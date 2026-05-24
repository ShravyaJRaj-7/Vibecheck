import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json());

interface Friend {
  name: string;
  phone: string;
}

interface Trip {
  id: string;
  name: string;
  friends: Friend[];
  preferences: any[];
  recommendations: any[];
  votes: any[];
  status: string;
  winner?: any;
}

const trips: Record<string, Trip> = {};
const adminStats = {
  totalRequests: 0,
  aiCosts: 0,
  latency: [] as number[],
  lastReset: Date.now(),
};

// Middleware for stats tracking
app.use((req, res, next) => {
  const start = Date.now();
  adminStats.totalRequests++;
  res.on("finish", () => {
    const duration = Date.now() - start;
    adminStats.latency.push(duration);
    if (adminStats.latency.length > 50) adminStats.latency.shift();
  });
  next();
});

// Gemini Setup
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY || "",
  httpOptions: {
    headers: {
      "User-Agent": "aistudio-build",
    },
  },
});

// API Routes
app.post("/api/trips", (req, res) => {
  const id = Math.random().toString(36).substring(2, 9);
  trips[id] = {
    id,
    name: req.body.name || "Untitled Trip",
    friends: req.body.friends || [],
    preferences: [],
    recommendations: [],
    votes: [],
    status: "preference_gathering", // survey -> recommendations -> voting -> consensus
  };
  res.json({ id });
});

app.get("/api/trips/:id", (req, res) => {
  const trip = trips[req.params.id];
  if (!trip) return res.status(404).json({ error: "Trip not found" });
  res.json(trip);
});

async function performRecommendation(trip: Trip) {
  if (trip.status !== "preference_gathering" && trip.status !== "generating") return;
  
  // Ensure status is generating while we work
  const previousStatus = trip.status;
  trip.status = "generating";
  
  adminStats.aiCosts += 0.002; 

  const prompt = `Analyze these travel preferences from a group of friends and recommend exactly 3 specific destinations (City, Country).
  For each destination:
  1. Provide a short 2-sentence reasoning why it fits the group.
  2. Provide a single descriptive keyword or short phrase (max 2 words) that captures the visual essence of this specific location for image searching (e.g., 'santorini sunset', 'tokyo neon', 'bali beach').
  
  Preferences:
  ${trip.preferences.map((p: any) => `${p.name}: ${p.text}`).join("\n")}
  
  Return the result in JSON format as an array of objects with 'destination', 'reason', and 'imageKeyword' fields.`;

  try {
    const result = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              destination: { type: Type.STRING },
              reason: { type: Type.STRING },
              imageKeyword: { type: Type.STRING },
            },
            required: ["destination", "reason", "imageKeyword"],
          },
        },
      },
    });

    const recommendations = JSON.parse(result.text || "[]");
    trip.recommendations = recommendations;
    trip.status = "voting";
  } catch (error) {
    console.error("Gemini Error:", error);
    // Reset status so user can retry if it got stuck
    trip.status = "preference_gathering";
  }
}

app.post("/api/trips/:id/preferences", async (req, res) => {
  const trip = trips[req.params.id];
  if (!trip) return res.status(404).json({ error: "Trip not found" });
  
  const { name, text } = req.body;
  
  // Prevent duplicate submissions from same person for now to keep logic simple
  if (!trip.preferences.some(p => p.name === name)) {
    trip.preferences.push({ name, text, timestamp: Date.now() });
  }

  // Check if everyone submitted
  const everyoneSubmitted = trip.friends.every(f => 
    trip.preferences.some(p => p.name === f.name)
  );

  if (everyoneSubmitted && trip.status === "preference_gathering") {
    // We trigger this async, we don't necessarily need to wait for it to return 200
    // but the next poll will see the "voting" status or "generating" status
    // To make it better, we can add a 'generating' status
    trip.status = "generating"; 
    performRecommendation(trip);
  }

  res.json({ status: "ok", tripStatus: trip.status });
});

app.post("/api/trips/:id/recommend", async (req, res) => {
  const trip = trips[req.params.id];
  if (!trip) return res.status(404).json({ error: "Trip not found" });

  if (trip.preferences.length === 0) {
    return res.status(400).json({ error: "No preferences gathered yet" });
  }

  await performRecommendation(trip);
  res.json(trip.recommendations);
});

app.post("/api/trips/:id/vote", (req, res) => {
  const trip = trips[req.params.id];
  if (!trip) return res.status(404).json({ error: "Trip not found" });

  const { name, rankings } = req.body; // rankings: [0, 1, 2] indices of recommendations
  trip.votes.push({ name, rankings });

  // Check if everyone voted
  if (trip.votes.length >= trip.friends.length) {
    // Ranked Choice Voting Logic (Simplified Borda Count for this demo)
    const scores = trip.recommendations.map(() => 0);
    trip.votes.forEach((v: any) => {
      // 1st gets 3 points, 2nd gets 2, 3rd gets 1
      v.rankings.forEach((targetIndex: number, rank: number) => {
        scores[targetIndex] += (3 - rank);
      });
    });

    let winnerIndex = scores.indexOf(Math.max(...scores));
    trip.winner = trip.recommendations[winnerIndex];
    trip.status = "completed";
  }

  res.json({ status: "ok" });
});

app.get("/api/admin/stats", (req, res) => {
  const avgLatency = adminStats.latency.length 
    ? adminStats.latency.reduce((a, b) => a + b, 0) / adminStats.latency.length 
    : 0;
    
  res.json({
    totalRequests: adminStats.totalRequests,
    aiCosts: adminStats.aiCosts.toFixed(4),
    avgLatency: avgLatency.toFixed(0) + "ms",
    activeTrips: Object.keys(trips).length,
    history: adminStats.latency.map((l, i) => ({ time: i, ms: l })),
  });
});

async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
