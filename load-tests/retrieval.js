import http from "k6/http";
import { check } from "k6";
import { Trend, Rate } from "k6/metrics";

const retrievalLatency = new Trend("retrieval_latency");
const failures = new Rate("retrieval_failures");

const vus = Number(__ENV.VUS || 25);
const duration = __ENV.DURATION || "5m";

export const options = {
  scenarios: {
    production_retrieval: {
      executor: "constant-vus",
      vus,
      duration,
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.01"],
    http_req_duration: ["p(95)<1000"],
    retrieval_failures: ["rate<0.01"],
  },
};

const questions = [
  "How is forecast bias calculated?",
  "What does WAPE mean?",
  "When is inventory considered a shortage risk?",
  "What controls apply to generated SQL?",
];

export default function () {
  const query = questions[Math.floor(Math.random() * questions.length)];
  const headers = {
    "Content-Type": "application/json",
  };

  if (__ENV.API_KEY) {
    headers.Authorization = `Bearer ${__ENV.API_KEY}`;
  }

  const response = http.post(
    `${__ENV.BASE_URL}/api/retrieve`,
    JSON.stringify({ query, limit: 5 }),
    { headers }
  );

  const successful = check(response, {
    "status is 200": (r) => r.status === 200,
    "contains results": (r) => {
      try {
        return JSON.parse(r.body).results.length > 0;
      } catch {
        return false;
      }
    },
  });

  failures.add(!successful);

  if (response.status === 200) {
    retrievalLatency.add(JSON.parse(response.body).retrieval_ms);
  }
}
