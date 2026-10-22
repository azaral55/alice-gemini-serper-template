import { randomUUID } from "node:crypto";
import { waitUntil } from "@vercel/functions";

const DEFAULT_MODEL = "gemini-3.5-flash-lite";
const MAX_HISTORY_MESSAGES = 8;
const MAX_HISTORY_MESSAGE_CHARS = 1200;
const MAX_ALICE_TEXT = 1024;
const TOTAL_RESPONSE_BUDGET_MS = 4100;
const SERPER_TIMEOUT_MS = 600;
const GEMINI_TIMEOUT_MS = 3500;
// With Redis connected, slow answers keep generating after Alice's deadline and are picked up on "дальше".
const BACKGROUND_GEMINI_TIMEOUT_MS = 25000;
const JOB_TTL_SECONDS = 600;
const REDIS_TIMEOUT_MS = 800;
const JOB_POLL_INTERVAL_MS = 300;

const THINKING_TEXT = "Дай подумать пару секунд. Потом скажи «дальше».";
const STILL_THINKING_TEXT = "Ещё думаю. Скажи «дальше» через пару секунд.";

const SYSTEM_PROMPT = [
  "Ты — полезный голосовой помощник в навыке Яндекс Алисы.",
  "Отвечай естественно, по-русски, кратко и так, чтобы ответ было удобно слушать.",
  "Обычно используй 1–3 коротких предложения. Не используй Markdown, таблицы, списки с символами или длинные вступления.",
  "Если вопрос требует актуальных данных и переданы результаты поиска, опирайся на них; не выдумывай отсутствующие факты.",
  "Если не знаешь ответа, честно скажи об этом."
].join(" ");

function aliceResponse(text, endSession = false, history = []) {
  return {
    version: "1.0",
    response: {
      text: fitAliceText(text),
      end_session: endSession
    },
    session_state: { history }
  };
}

function fitAliceText(value) {
  const text = String(value ?? "")
    .replace(/[\r\n\t]+/g, " ")
    .replace(/\s{2,}/g, " ")
    .trim();

  if (text.length <= MAX_ALICE_TEXT) return text || "Не удалось подготовить ответ.";

  const candidate = text.slice(0, MAX_ALICE_TEXT - 1);
  const sentenceEnd = Math.max(
    candidate.lastIndexOf(". "),
    candidate.lastIndexOf("! "),
    candidate.lastIndexOf("? ")
  );
  if (sentenceEnd >= 300) return candidate.slice(0, sentenceEnd + 1).trim();

  const lastSpace = candidate.lastIndexOf(" ");
  return (lastSpace > 200 ? candidate.slice(0, lastSpace) : candidate).trim() + "…";
}

const modelsWithoutThinkingConfig = new Set();

// Thinking makes Gemini answer too slowly for Alice, so ask for the smallest amount the model allows.
function thinkingConfigFor(model) {
  if (/^gemini-2\.5-flash/.test(model)) return { thinkingBudget: 0 };
  if (/^gemini-[3-9]/.test(model)) return { thinkingLevel: /pro/.test(model) ? "low" : "minimal" };
  return undefined;
}

function isSearchIntent(text) {
  return /\b(latest|today|current|news|weather|price|exchange rate|recent|search|look up)\b|сегодня|сейчас|последн|новост|погод|курс валют|найди|поищи|актуальн|сколько стоит|свежие данные/i.test(text);
}

function safeHistory(history) {
  if (!Array.isArray(history)) return [];
  return history
    .filter(item => item && (item.role === "user" || item.role === "model") && typeof item.text === "string")
    .slice(-MAX_HISTORY_MESSAGES)
    .map(item => ({ role: item.role, text: item.text.slice(0, MAX_HISTORY_MESSAGE_CHARS) }));
}

async function fetchJson(url, options, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), Math.max(100, timeoutMs));
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    const data = await response.json().catch(() => ({}));
    return { response, data };
  } finally {
    clearTimeout(timer);
  }
}

async function searchWeb(query, timeoutMs) {
  const key = process.env.SERPER_API_KEY;
  if (!key || timeoutMs < 150) return "";

  try {
    const { response, data } = await fetchJson(
      "https://google.serper.dev/search",
      {
        method: "POST",
        headers: {
          "X-API-KEY": key,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ q: query, gl: "ru", hl: "ru", num: 3 })
      },
      Math.min(SERPER_TIMEOUT_MS, timeoutMs)
    );

    if (!response.ok) {
      console.warn("Serper request failed with HTTP", response.status);
      return "";
    }

    const items = (Array.isArray(data.organic) ? data.organic : [])
      .slice(0, 3)
      .map((item, i) => {
        const title = String(item.title || "").slice(0, 180);
        const snippet = String(item.snippet || "").slice(0, 350);
        const link = String(item.link || "").slice(0, 240);
        return `${i + 1}. ${title}: ${snippet} ${link}`.trim();
      });

    const answerBox = data.answerBox?.answer || data.answerBox?.snippet;
    if (answerBox) items.unshift(`Краткий ответ поисковика: ${String(answerBox).slice(0, 350)}`);
    return items.length ? items.join("\n") : "";
  } catch (error) {
    console.warn("Serper request did not finish in time; continuing without search.");
    return "";
  }
}

async function askGemini(text, history, searchContext, timeoutMs) {
  const key = process.env.GEMINI_API_KEY;
  if (!key) throw new Error("MISSING_GEMINI_KEY");
  if (timeoutMs < 250) throw new Error("TIME_BUDGET_EXHAUSTED");

  const model = String(process.env.GEMINI_MODEL || DEFAULT_MODEL).trim();
  if (!/^[A-Za-z0-9._-]+$/.test(model)) throw new Error("INVALID_MODEL_NAME");

  const contents = safeHistory(history).map(item => ({
    role: item.role,
    parts: [{ text: item.text }]
  }));

  const prompt = searchContext
    ? `${text}\n\nРезультаты поиска в интернете:\n${searchContext}\nИспользуй их только там, где они относятся к вопросу.`
    : text;
  contents.push({ role: "user", parts: [{ text: prompt }] });

  const startedAt = Date.now();
  const callGemini = thinkingConfig => streamGemini(
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:streamGenerateContent?alt=sse`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-goog-api-key": key
      },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] },
        contents,
        generationConfig: {
          temperature: 0.5,
          maxOutputTokens: 240,
          ...(thinkingConfig && { thinkingConfig })
        }
      })
    },
    timeoutMs - (Date.now() - startedAt)
  );

  const thinkingConfig = modelsWithoutThinkingConfig.has(model) ? undefined : thinkingConfigFor(model);
  let result = await callGemini(thinkingConfig);
  if (result.status === 400 && thinkingConfig) {
    // Some models reject the thinking setting; retry once with the model's defaults
    // and skip the setting on later requests so they don't pay for a second call.
    console.warn("Gemini rejected thinkingConfig; retrying without it.");
    modelsWithoutThinkingConfig.add(model);
    result = await callGemini(undefined);
  }

  if (result.timedOut && !result.text.trim()) {
    console.warn(`Gemini (${model}) sent nothing within ${Date.now() - startedAt} ms.`);
    throw new Error("TIME_BUDGET_EXHAUSTED");
  }

  if (!result.ok) {
    const status = Number(result.status);
    console.error("Gemini request failed with HTTP", status);
    if (status === 401 || status === 403) throw new Error("GEMINI_KEY_REJECTED");
    if (status === 429) throw new Error("GEMINI_QUOTA");
    throw new Error("GEMINI_REQUEST_FAILED");
  }

  const answer = result.text.trim();
  if (!answer) {
    if (result.blocked) throw new Error("GEMINI_BLOCKED");
    throw new Error("GEMINI_EMPTY_RESPONSE");
  }
  if (result.timedOut) {
    // Alice would drop a late reply entirely, so answer with what Gemini managed to say in time.
    console.warn("Gemini did not finish in time; returning the partial answer.");
    return cutAtSentence(answer);
  }
  if (result.finishReason === "MAX_TOKENS") {
    console.warn("Gemini reached the output token limit.");
  }
  console.log(`Gemini (${model}) answered in ${Date.now() - startedAt} ms.`);
  return answer;
}

// Reads Gemini's server-sent events, keeping the text received so far if the time budget runs out.
async function streamGemini(url, options, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), Math.max(100, timeoutMs));
  const result = { ok: false, status: 0, text: "", timedOut: false, blocked: false, finishReason: "" };
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    result.ok = response.ok;
    result.status = response.status;
    if (!response.ok) return result;

    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value;
      let newline;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (!line.startsWith("data:")) continue;
        let chunk;
        try {
          chunk = JSON.parse(line.slice(5));
        } catch {
          continue;
        }
        const candidate = chunk?.candidates?.[0];
        for (const part of candidate?.content?.parts || []) {
          if (typeof part.text === "string" && !part.thought) result.text += part.text;
        }
        if (candidate?.finishReason) result.finishReason = candidate.finishReason;
        if (chunk?.promptFeedback?.blockReason) result.blocked = true;
      }
    }
    return result;
  } catch (error) {
    if (error?.name !== "AbortError") throw error;
    result.timedOut = true;
    return result;
  } finally {
    clearTimeout(timer);
  }
}

function cutAtSentence(value) {
  const text = String(value).trim();
  const sentenceEnd = Math.max(text.lastIndexOf(". "), text.lastIndexOf("! "), text.lastIndexOf("? "));
  if (/[.!?]$/.test(text)) return text;
  if (sentenceEnd >= 10) return text.slice(0, sentenceEnd + 1);
  return text + "…";
}

function errorForUser(error) {
  switch (error?.message) {
    case "MISSING_GEMINI_KEY":
      return "Сервер пока не настроен. Проверьте переменную GEMINI_API_KEY в настройках Vercel.";
    case "GEMINI_KEY_REJECTED":
      return "Не удалось подключиться к Gemini. Проверьте API-ключ в настройках Vercel.";
    case "GEMINI_QUOTA":
      return "Сейчас превышен лимит запросов к Gemini. Попробуйте немного позже.";
    case "GEMINI_BLOCKED":
      return "Я не смогла обработать этот запрос. Попробуйте спросить иначе.";
    case "TIME_BUDGET_EXHAUSTED":
      return "Не успела подготовить ответ. Попробуйте задать вопрос ещё раз, чуть короче.";
    default:
      return "Не получилось получить ответ от нейросети. Попробуйте ещё раз чуть позже.";
  }
}

function isContinueCommand(text) {
  const normalized = String(text).toLowerCase().replace(/ё/g, "е").replace(/[^a-zа-я ]+/g, " ").replace(/\s+/g, " ").trim();
  return /^(алиса )?(дальше|ну|ну что|ну и|ну и что|и что|что там|ответ|готово|готов|продолжай|продолжи|давай|ну давай)$/.test(normalized);
}

function redisConfig() {
  const url = process.env.KV_REST_API_URL || process.env.UPSTASH_REDIS_REST_URL;
  const token = process.env.KV_REST_API_TOKEN || process.env.UPSTASH_REDIS_REST_TOKEN;
  return url && token ? { url: url.replace(/\/+$/, ""), token } : null;
}

async function redis(command) {
  const config = redisConfig();
  const { response, data } = await fetchJson(
    config.url,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${config.token}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify(command)
    },
    REDIS_TIMEOUT_MS
  );
  if (!response.ok || data?.error) throw new Error("REDIS_FAILED");
  return data.result;
}

async function readJob(key) {
  const value = await redis(["GET", key]);
  if (!value) return null;
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function writeJob(key, job) {
  return redis(["SET", key, JSON.stringify(job), "EX", String(JOB_TTL_SECONDS)]);
}

// Stores the background result unless the user has already asked something newer.
async function finishJob(key, id, update) {
  const current = await readJob(key);
  if (current?.id !== id) return;
  await writeJob(key, { ...current, ...update });
}

async function waitForJob(key, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let job = await readJob(key);
  while (job?.status === "pending" && Date.now() + JOB_POLL_INTERVAL_MS < deadline) {
    await new Promise(resolve => setTimeout(resolve, JOB_POLL_INTERVAL_MS));
    job = await readJob(key);
  }
  return job;
}

function withHistory(history, question, answer) {
  return [
    ...history,
    { role: "user", text: question.slice(0, MAX_HISTORY_MESSAGE_CHARS) },
    { role: "model", text: answer.slice(0, MAX_HISTORY_MESSAGE_CHARS) }
  ].slice(-MAX_HISTORY_MESSAGES);
}

function timeout(ms) {
  return new Promise(resolve => setTimeout(() => resolve(null), Math.max(0, ms)));
}

export default async function handler(req, res) {
  if (req.method === "GET") {
    return res.status(200).json({
      ok: true,
      service: "alice-gemini-serper",
      message: "Webhook is ready. Yandex Dialogs should send POST requests to /api."
    });
  }

  if (req.method !== "POST") {
    res.setHeader("Allow", "GET, POST");
    return res.status(405).json({ error: "Method not allowed" });
  }

  const startedAt = Date.now();
  const remainingMs = () => TOTAL_RESPONSE_BUDGET_MS - (Date.now() - startedAt);
  const body = req.body && typeof req.body === "object" ? req.body : {};
  const text = String(body?.request?.command || body?.request?.original_utterance || "").trim();
  const oldHistory = safeHistory(body?.state?.session?.history);

  if (!text) {
    return res.status(200).json(aliceResponse(
      "Привет! Я могу ответить на вопрос или найти актуальную информацию. Что хочешь узнать?",
      false,
      oldHistory
    ));
  }

  const sessionId = String(body?.session?.session_id || "");
  const jobKey = redisConfig() && sessionId ? `alice:job:${sessionId}` : null;

  try {
    if (jobKey && isContinueCommand(text)) {
      const job = await waitForJob(jobKey, remainingMs() - 600).catch(error => {
        console.warn("Could not read the background answer:", error.message);
        return null;
      });
      if (job?.status === "pending") {
        return res.status(200).json(aliceResponse(STILL_THINKING_TEXT, false, oldHistory));
      }
      if (job?.status === "done") {
        await redis(["DEL", jobKey]).catch(() => {});
        return res.status(200).json(aliceResponse(job.answer, false, withHistory(oldHistory, job.question, job.answer)));
      }
      if (job?.status === "error") {
        await redis(["DEL", jobKey]).catch(() => {});
        return res.status(200).json(aliceResponse(errorForUser(new Error(job.error)), false, oldHistory));
      }
      // Nothing is waiting, so treat the phrase as an ordinary question.
    }

    let searchContext = "";
    if (isSearchIntent(text) && process.env.SERPER_API_KEY) {
      // Search has a very short budget because Alice expects a response within 4.5 seconds.
      searchContext = await searchWeb(text, Math.min(SERPER_TIMEOUT_MS, Math.max(0, remainingMs() - 2600)));
    }

    const quickBudgetMs = Math.max(0, Math.min(GEMINI_TIMEOUT_MS, remainingMs() - 600));

    if (!jobKey) {
      const answer = await askGemini(text, oldHistory, searchContext, quickBudgetMs);
      return res.status(200).json(aliceResponse(answer, false, withHistory(oldHistory, text, answer)));
    }

    const geminiAnswer = askGemini(text, oldHistory, searchContext, BACKGROUND_GEMINI_TIMEOUT_MS);
    const quickAnswer = await Promise.race([geminiAnswer, timeout(quickBudgetMs)]);
    if (quickAnswer !== null) {
      return res.status(200).json(aliceResponse(quickAnswer, false, withHistory(oldHistory, text, quickAnswer)));
    }

    // Gemini needs more time than Alice allows: keep it running and hand the answer over on "дальше".
    const id = randomUUID();
    try {
      await writeJob(jobKey, { id, status: "pending", question: text });
    } catch (error) {
      console.warn("Could not save the pending answer:", error.message);
      throw new Error("TIME_BUDGET_EXHAUSTED");
    }
    console.log("Gemini is still answering; continuing in the background.");
    waitUntil(
      geminiAnswer
        .then(answer => finishJob(jobKey, id, { status: "done", answer }))
        .catch(error => finishJob(jobKey, id, { status: "error", error: error?.message || "GEMINI_REQUEST_FAILED" }))
        .catch(error => console.warn("Could not save the background answer:", error.message))
    );
    return res.status(200).json(aliceResponse(THINKING_TEXT, false, oldHistory));
  } catch (error) {
    const safeError = error?.name === "AbortError" ? new Error("TIME_BUDGET_EXHAUSTED") : error;
    console.warn("Alice webhook returned a fallback response:", safeError?.message || "unknown error");
    return res.status(200).json(aliceResponse(errorForUser(safeError), false, oldHistory));
  }
}

export { aliceResponse, cutAtSentence, fitAliceText, isContinueCommand, isSearchIntent, safeHistory, thinkingConfigFor };
