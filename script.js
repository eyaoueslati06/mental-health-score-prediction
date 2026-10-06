const API_URL = "http://127.0.0.1:8000/predict";
const MAX_SCORE = 10; // score scale shown in the gauge
const CIRCUMFERENCE = 2 * Math.PI * 54;

const form = document.getElementById("predict-form");
const submitBtn = document.getElementById("submit-btn");
const resetBtn = document.getElementById("reset-btn");
const errorBox = document.getElementById("error");
const card = document.getElementById("result-card");
const gaugeBar = document.getElementById("gauge-bar");
const scoreValue = document.getElementById("score-value");
const scoreTitle = document.getElementById("score-title");
const scoreText = document.getElementById("score-text");
const resultState = card.querySelector(".state-result");

const INT_FIELDS = ["age", "daily_unlocks"];
const FLOAT_FIELDS = [
  "avg_daily_usage_hours",
  "study_hours",
  "physical_activity_hours",
  "sleep_hours_per_night",
];

function setState(state) {
  card.dataset.state = state;
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function clearError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
  form.querySelectorAll(".invalid").forEach((el) => el.classList.remove("invalid"));
}

/* Check every field against the backend limits before sending. */
function validate(data) {
  const problems = [];
  const mark = (name) => {
    const el = form.elements[name];
    const target = el instanceof RadioNodeList ? el[0].closest(".segmented") : el;
    target.classList.add("invalid");
  };

  const rules = [
    ["age", (v) => Number.isInteger(v) && v >= 10 && v <= 100, "Age must be between 10 and 100."],
    ["avg_daily_usage_hours", (v) => v >= 0 && v <= 24, "Daily usage must be between 0 and 24 hours."],
    ["study_hours", (v) => v >= 0 && v <= 24, "Study hours must be between 0 and 24."],
    ["physical_activity_hours", (v) => v >= 0 && v <= 24, "Exercise hours must be between 0 and 24."],
    ["sleep_hours_per_night", (v) => v >= 0 && v <= 24, "Sleep hours must be between 0 and 24."],
    ["daily_unlocks", (v) => Number.isInteger(v) && v >= 0, "Unlocks must be a whole number of 0 or more."],
  ];

  rules.forEach(([name, ok, msg]) => {
    const v = data[name];
    if (typeof v !== "number" || Number.isNaN(v) || !ok(v)) {
      problems.push(msg);
      mark(name);
    }
  });

  ["gender", "country", "academic_level", "most_used_platform", "purpose_of_use", "stress_level"].forEach((name) => {
    if (!data[name]) {
      problems.push("Please complete all fields.");
      mark(name);
    }
  });

  return [...new Set(problems)];
}

function collectData() {
  const raw = Object.fromEntries(new FormData(form).entries());
  const data = { ...raw };
  if (data.country) data.country = data.country.trim();
  INT_FIELDS.forEach((k) => (data[k] = raw[k] === "" || raw[k] === undefined ? NaN : Number(raw[k])));
  FLOAT_FIELDS.forEach((k) => (data[k] = raw[k] === "" || raw[k] === undefined ? NaN : Number(raw[k])));
  return data;
}

/* Turn FastAPI's 422 error list into a readable message. */
function formatApiError(body) {
  if (body && Array.isArray(body.detail)) {
    return body.detail
      .map((d) => `${(d.loc || []).slice(1).join(" ")}: ${d.msg}`)
      .join(" | ");
  }
  return "The server could not process this request.";
}

function describe(score) {
  const ratio = score / MAX_SCORE;
  if (ratio >= 0.7) return { tone: "good", color: "var(--good)", title: "Doing well", text: "Your habits point to a healthy mental wellbeing score. Keep up the routines that work for you." };
  if (ratio >= 0.5) return { tone: "mid", color: "var(--mid)", title: "Room to improve", text: "Your score is moderate. Small changes to sleep, screen time or exercise could help." };
  return { tone: "low", color: "var(--low)", title: "Needs attention", text: "Your score is on the low side. Consider cutting screen time, sleeping more and talking to someone you trust." };
}

function animateNumber(target, duration = 1200) {
  const start = performance.now();
  const step = (now) => {
    const t = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    scoreValue.textContent = (target * eased).toFixed(1);
    if (t < 1) requestAnimationFrame(step);
    else scoreValue.textContent = target.toFixed(1);
  };
  requestAnimationFrame(step);
}

function showResult(score) {
  const info = describe(score);
  const fraction = Math.max(0, Math.min(score / MAX_SCORE, 1));

  resultState.classList.remove("tone-good", "tone-mid", "tone-low");
  resultState.classList.add(`tone-${info.tone}`);
  scoreTitle.textContent = info.title;
  scoreText.textContent = info.text;
  gaugeBar.style.stroke = info.color;
  gaugeBar.style.strokeDashoffset = CIRCUMFERENCE;
  scoreValue.textContent = "0.0";

  setState("result");

  // Start the animations on the next frame so the transition runs.
  requestAnimationFrame(() => {
    gaugeBar.style.strokeDashoffset = CIRCUMFERENCE * (1 - fraction);
    animateNumber(score);
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();

  const data = collectData();
  const problems = validate(data);
  if (problems.length) {
    showError(problems.join(" "));
    return;
  }

  submitBtn.disabled = true;
  submitBtn.textContent = "Predicting...";
  setState("loading");
  if (window.matchMedia("(max-width: 900px)").matches) {
    card.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    const body = await response.json().catch(() => null);
    if (!response.ok) throw new Error(formatApiError(body));

    showResult(body.predicted_mental_health_score);
  } catch (err) {
    setState("idle");
    const offline = err instanceof TypeError;
    showError(
      offline
        ? "Cannot reach the server. Check that FastAPI is running at 127.0.0.1:8000."
        : err.message
    );
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Predict my score";
  }
});

resetBtn.addEventListener("click", () => {
  form.reset();
  clearError();
  setState("idle");
  form.scrollIntoView({ behavior: "smooth", block: "start" });
});

form.addEventListener("input", (e) => {
  e.target.classList.remove("invalid");
  e.target.closest(".segmented")?.classList.remove("invalid");
});