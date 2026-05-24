import { useState, useEffect } from "react";
import { analyzeGroupSynergy } from "./lib/ml_engine";
import { 
  Plane, 
  MapPin, 
  Users, 
  Send, 
  Vote, 
  Settings, 
  History, 
  CheckCircle2, 
  MessageCircle,
  Copy,
  TrendingDown,
  Activity,
  Cpu,
  DollarSign,
  Heart,
  Cloud,
  Luggage,
  Sun,
  Camera,
  Compass,
  Star
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { 
  AreaChart,
  Area,
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer
} from "recharts";

// --- Types ---
interface Preference {
  name: string;
  text: string;
  timestamp: number;
}

interface Recommendation {
  destination: string;
  reason: string;
  imageKeyword?: string;
}

interface VoteData {
  name: string;
  rankings: number[];
}

interface Friend {
  name: string;
  phone: string;
}

interface Trip {
  id: string;
  name: string;
  friends: Friend[];
  preferences: Preference[];
  recommendations: Recommendation[];
  votes: VoteData[];
  status: "preference_gathering" | "generating" | "voting" | "completed";
  winner?: Recommendation;
}

interface AdminStats {
  totalRequests: number;
  aiCosts: string;
  avgLatency: string;
  activeTrips: number;
  history: { time: number; ms: number }[];
}

// --- Components ---

const FloatingDecor = () => (
  <div className="absolute inset-0 overflow-hidden pointer-events-none z-0">
    <motion.div 
      animate={{ y: [0, -20, 0], x: [0, 10, 0] }}
      transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }}
      className="absolute top-[10%] left-[5%] opacity-[0.08]"
    >
      <Cloud className="w-48 h-48 text-teal" />
    </motion.div>
    <motion.div 
      animate={{ y: [0, 30, 0], x: [0, -15, 0] }}
      transition={{ duration: 12, repeat: Infinity, ease: "easeInOut" }}
      className="absolute top-[40%] right-[5%] opacity-[0.08]"
    >
      <Cloud className="w-64 h-64 text-coral" />
    </motion.div>
    <motion.div 
      animate={{ rotate: 360 }}
      transition={{ duration: 60, repeat: Infinity, ease: "linear" }}
      className="absolute bottom-[10%] left-[10%] opacity-[0.05]"
    >
      <Compass className="w-32 h-32 text-dark-ink" />
    </motion.div>
    <motion.div 
      animate={{ scale: [1, 1.2, 1], opacity: [0.1, 0.2, 0.1] }}
      transition={{ duration: 4, repeat: Infinity }}
      className="absolute top-[20%] right-[15%] opacity-[0.1]"
    >
      <Star className="w-12 h-12 text-mustard fill-mustard" />
    </motion.div>
    <motion.div 
      animate={{ x: [-100, 2000] }}
      transition={{ duration: 30, repeat: Infinity, ease: "linear" }}
      className="absolute top-[30%] -left-20 opacity-[0.05]"
    >
      <Plane className="w-32 h-32 text-dark-ink -rotate-12" />
    </motion.div>
  </div>
);

const Header = () => (
  <header className="py-12 px-6 text-center relative overflow-hidden z-20">
    <div className="absolute top-0 left-0 w-full h-full vintage-overlay pointer-events-none" />
    <motion.div 
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="inline-block relative"
    >
      <div className="absolute -top-12 -right-16 rotate-12 opacity-90 hidden sm:block">
        <div className="post-stamp flex flex-col items-center gap-1 p-3 shadow-lg">
          <Plane className="text-coral w-6 h-6" />
          <span className="text-[8px] font-mono font-bold leading-none">AIR MAIL</span>
        </div>
      </div>
      <div className="absolute -bottom-8 -left-12 -rotate-12 opacity-90 hidden sm:block">
        <div className="sticker flex items-center gap-1 px-4 py-1.5 bg-white scale-110">
          <Camera className="w-3 h-3 text-teal" />
          <span>EST. 2026</span>
        </div>
      </div>
      <h1 className="text-5xl sm:text-8xl font-sans mb-3 tracking-tighter drop-shadow-sm text-dark-ink">
        VibeCheck <span className="text-coral">Travel</span>
      </h1>
      <div className="flex items-center justify-center gap-4">
        <div className="h-[2px] w-12 bg-mustard/40" />
        <p className="text-teal font-mono text-xs sm:text-sm uppercase tracking-[0.4em] font-bold">Analog Aesthetics • Digital Decisions</p>
        <div className="h-[2px] w-12 bg-mustard/40" />
      </div>
    </motion.div>
  </header>
);

const MLGroupAnalytics = ({ friends, preferences }: { friends: any[]; preferences: any[] }) => {
  if (preferences.length === 0) {
    return (
      <div className="retro-card p-6 bg-white space-y-4">
        <h3 className="text-xs uppercase font-bold tracking-widest flex items-center gap-2 text-dark-ink">
          <Cpu className="w-4 h-4 text-coral" /> NLP Vibe Engine
        </h3>
        <p className="text-xs text-dark-ink/60 italic">
          Waiting for passengers to stamp their travel wishes to build the high-dimensional Vector Space...
        </p>
      </div>
    );
  }

  const { matrix, averageSynergy, strongestPair, topWeights } = analyzeGroupSynergy(friends, preferences);

  return (
    <div className="retro-card p-6 bg-white space-y-6 relative overflow-hidden">
      <div className="retro-tape bg-mustard/30" />
      <div className="flex justify-between items-center">
        <h3 className="text-xs uppercase font-bold tracking-widest flex items-center gap-2 text-dark-ink">
          <Cpu className="w-4 h-4 text-teal animate-pulse" /> NLP Vibe Engine v2.5
        </h3>
        <span className="text-[10px] font-mono font-bold bg-dark-ink/5 border border-dark-ink/10 px-2 py-0.5 rounded text-dark-ink/50">
          TF-IDF + Cosine
        </span>
      </div>

      {preferences.length < 2 ? (
        <div className="space-y-3">
          <div className="bg-vanilla/25 p-3 rounded-lg border border-dark-ink/5 text-xs text-dark-ink/75">
            🎙️ <strong className="font-bold">Gathering corpus data...</strong> Submit at least <strong className="font-bold">2 people's</strong> stamps to calculate compatibility matching vectors.
          </div>
          {preferences.length === 1 && (
            <div className="space-y-1">
              <div className="text-[10px] font-bold uppercase text-teal">Extracted features from {preferences[0].name}:</div>
              <div className="flex flex-wrap gap-1.5">
                {topWeights.map((tw, i) => (
                  <span key={i} className="text-xs bg-teal/10 text-teal px-2 py-0.5 rounded font-mono font-bold">
                    {tw.word}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="space-y-6">
          {/* Synergy Index */}
          <div className="flex items-center justify-between p-3 bg-vanilla/35 rounded-lg border-2 border-dark-ink/10">
            <div>
              <div className="text-[10px] font-bold uppercase tracking-wide text-dark-ink/50">Group Compatibility</div>
              <div className="text-3xl font-sans text-dark-ink font-extrabold flex items-baseline gap-1">
                {(averageSynergy * 100).toFixed(0)}%
                <span className="text-xs font-mono font-normal text-dark-ink/40">alignment</span>
              </div>
            </div>
            <div className="text-right">
              <div className="text-[10px] font-bold uppercase tracking-wide text-dark-ink/50">Strongest Synergy</div>
              <div className="text-xs font-bold text-teal font-sans">
                {strongestPair ? `${strongestPair.friend1} 🤝 ${strongestPair.friend2}` : 'None'}
              </div>
              <div className="text-[10px] text-dark-ink/40 font-mono">
                {strongestPair ? `${(strongestPair.similarity * 100).toFixed(0)}% cosine match` : ''}
              </div>
            </div>
          </div>

          {/* Interactive Heatmap Matrix */}
          <div className="space-y-2">
            <h4 className="text-[10px] font-bold uppercase tracking-widest text-teal">Cosine Similarity Heatmap</h4>
            <div className="overflow-x-auto rounded-lg border-2 border-dark-ink/10 bg-white">
              <table className="min-w-full text-[11px] font-mono border-collapse">
                <thead>
                  <tr className="bg-vanilla/40 border-b border-dark-ink/10">
                    <th className="p-2 border-r border-dark-ink/10 text-left text-dark-ink/60 font-medium">Friend</th>
                    {friends.map((f, i) => (
                      <th key={i} className="p-2 text-center text-dark-ink/70 font-bold max-w-[60px] truncate">{f.name}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {friends.map((f1, rIdx) => {
                    const wishesSubmitted = preferences.some(p => p.name === f1.name);
                    return (
                      <tr key={rIdx} className="border-b border-dark-ink/10 last:border-b-0 hover:bg-vanilla/10">
                        <td className="p-2 border-r border-dark-ink/10 font-bold text-dark-ink text-left truncate max-w-[80px]">
                          {f1.name}
                          {!wishesSubmitted && <span className="block text-[8px] font-normal text-dark-ink/30 italic">No stamp</span>}
                        </td>
                        {friends.map((f2, cIdx) => {
                          const val = matrix[f1.name]?.[f2.name] ?? 0.0;
                          const formattedVal = wishesSubmitted && preferences.some(p => p.name === f2.name) 
                            ? `${(val * 100).toFixed(0)}%` 
                            : '—';
                          
                          // Style based on score value
                          let bgClass = "bg-white";
                          let textClass = "text-dark-ink/30";
                          if (formattedVal !== '—') {
                            textClass = "text-dark-ink font-bold";
                            if (val > 0.8) bgClass = "bg-teal/40";
                            else if (val > 0.5) bgClass = "bg-teal/20";
                            else if (val > 0.2) bgClass = "bg-teal/5";
                            else if (val > 0.0) bgClass = "bg-coral/5 text-coral/80";
                            else bgClass = "bg-vanilla/20 text-dark-ink/50";
                          }

                          return (
                            <td key={cIdx} className={`p-2 text-center ${bgClass} ${textClass} border-r border-dark-ink/10 last:border-r-0 font-medium transition-all`}>
                              {formattedVal}
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Highest TF-IDF Word Weights */}
          <div className="space-y-2">
            <h4 className="text-[10px] font-bold uppercase tracking-widest text-teal">Extracted Core Interests (TF-IDF Weighting)</h4>
            <div className="flex flex-wrap gap-1.5">
              {topWeights.map((tw, i) => (
                <div key={i} className="flex items-center gap-1.5 bg-vanilla/50 rounded-full px-3 py-1 border border-dark-ink/10 hover:border-teal transition-all">
                  <span className="text-xs font-semibold text-dark-ink/80">{tw.word}</span>
                  <div className="h-2 w-10 bg-dark-ink/10 rounded-full overflow-hidden">
                    <div className="h-full bg-teal" style={{ width: `${Math.min(100, tw.weight * 130)}%` }} />
                  </div>
                  <span className="text-[9px] font-mono text-dark-ink/40">{(tw.weight * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Mini explanation badge */}
      <div className="text-[9px] opacity-50 bg-vanilla p-2.5 rounded border border-dark-ink/5 leading-relaxed text-dark-ink/70">
        💡 <strong className="font-bold">How it works:</strong> Freeform texts are analyzed using standard NLP tokenizers and stopword filters, then vectorized into Term Frequency - Inverse Document Frequency (TF-IDF) feature spaces. Cosine Similarity (vector dot-product over magnitudes) evaluates vibe alignment between group members prior to generating recommendations with Gemini Generative AI.
      </div>
    </div>
  );
};

export default function App() {
  const [view, setView] = useState<"home" | "trip" | "admin">("home");
  const [isPassenger, setIsPassenger] = useState(false);
  const [tripId, setTripId] = useState<string | null>(null);
  const [trip, setTrip] = useState<Trip | null>(null);
  const [adminStats, setAdminStats] = useState<AdminStats | null>(null);
  const [userName, setUserName] = useState("");
  const [loading, setLoading] = useState(false);

  // New Trip Form State
  const [newTripName, setNewTripName] = useState("");
  const [friendsInputs, setFriendsInputs] = useState<Friend[]>([{ name: "", phone: "" }]);

  // Preference Form State
  const [prefText, setPrefText] = useState("");

  // Voting State
  const [voteRankings, setVoteRankings] = useState<number[]>([]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const id = params.get("id");
    if (id) {
      setTripId(id);
      setIsPassenger(true);
      setView("trip");
    }
  }, []);

  useEffect(() => {
    if (tripId) {
      const pollRate = trip?.status === "generating" ? 1000 : 3000;
      const interval = setInterval(fetchTrip, pollRate);
      return () => clearInterval(interval);
    }
  }, [tripId, trip?.status]);

  useEffect(() => {
    if (view === "admin") {
      fetchAdminStats();
      const interval = setInterval(fetchAdminStats, 5000);
      return () => clearInterval(interval);
    }
  }, [view]);

  const fetchTrip = async () => {
    if (!tripId) return;
    try {
      const res = await fetch(`/api/trips/${tripId}`);
      const data = await res.json();
      setTrip(data);
    } catch (e) {
      console.error(e);
    }
  };

  const fetchAdminStats = async () => {
    try {
      const res = await fetch("/api/admin/stats");
      const data = await res.json();
      setAdminStats(data);
    } catch (e) {
      console.error(e);
    }
  };

  const createTrip = async () => {
    if (!newTripName || friendsInputs.some(f => !f.name)) return;
    setLoading(true);
    try {
      const validFriends = friendsInputs.filter(f => f.name.trim());
      const res = await fetch("/api/trips", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: newTripName, friends: validFriends }),
      });
      const data = await res.json();
      setTripId(data.id);
      window.history.pushState({}, "", `?id=${data.id}`);
      setView("trip");
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const submitPreference = async () => {
    if (!userName || !prefText || !tripId) return;
    setLoading(true);
    try {
      const res = await fetch(`/api/trips/${tripId}/preferences`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: userName, text: prefText }),
      });
      const data = await res.json();
      setPrefText("");
      
      // If server says we've moved to generating, update status immediately
      if (data.tripStatus === "generating") {
        setTrip(prev => prev ? { ...prev, status: "generating" } : null);
      } else {
        await fetchTrip();
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const triggerAI = async () => {
    if (!tripId) return;
    setLoading(true);
    try {
      await fetch(`/api/trips/${tripId}/recommend`, { method: "POST" });
      fetchTrip();
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const submitVote = async () => {
    if (!userName || voteRankings.length < 3 || !tripId) return;
    setLoading(true);
    try {
      await fetch(`/api/trips/${tripId}/vote`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: userName, rankings: voteRankings }),
      });
      setVoteRankings([]);
      fetchTrip();
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const toggleVoteSelection = (index: number) => {
    setVoteRankings(prev => {
      if (prev.includes(index)) {
        return prev.filter(i => i !== index);
      }
      if (prev.length >= 3) return prev;
      return [...prev, index];
    });
  };

  const generateWhatsAppLink = (phoneNumber: string) => {
    const text = encodeURIComponent(`✈️ Help us decide our "${trip?.name}" trip! Add your vibe and vote here: ${window.location.origin}/?id=${trip?.id}`);
    return `https://wa.me/${phoneNumber.replace(/\D/g, "")}?text=${text}`;
  };

  // --- Main Render ---

  return (
    <div className="min-h-screen relative selection:bg-teal selection:text-white font-body bg-vanilla overflow-x-hidden">
      {/* Global Texture Overlay */}
      <div className="fixed inset-0 vintage-overlay pointer-events-none z-50 opacity-[0.15]" />
      
      <FloatingDecor />
      
      <div className="relative z-10 flex flex-col items-center pb-20 px-6">
        {view !== "admin" && <Header />}
        
        <AnimatePresence mode="wait">
          {view === "home" && (
            <motion.div 
              key="home"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 1.05 }}
              className="w-full max-w-lg mt-8"
            >
              <div className="retro-card p-8 space-y-6 bg-white relative">
                <div className="retro-tape" />
                <div className="absolute top-4 right-4 rotate-12">
                   <div className="sticker bg-coral text-white border-white">EST. 2026</div>
                </div>
                
                <div className="text-center space-y-2 pt-4">
                  <div className="flex justify-center mb-4">
                    <div className="p-4 bg-vanilla rounded-full border-2 border-dark-ink group">
                      <Luggage className="w-8 h-8 text-teal group-hover:rotate-12 transition-transform" />
                    </div>
                  </div>
                  <h2 className="text-3xl font-sans text-dark-ink">New Adventure</h2>
                  <p className="text-dark-ink/60 text-sm italic italic">Step 1: Gather your troop</p>
                </div>

                <div className="space-y-4">
                  <div className="space-y-1">
                    <label className="text-xs uppercase font-bold tracking-widest text-teal ml-1">Trip Name</label>
                    <input 
                      type="text" 
                      placeholder="e.g. Summer Bash 2026"
                      className="w-full bg-vanilla/30 border-2 border-dark-ink/20 rounded-lg px-4 py-3 focus:border-teal outline-none transition-colors text-dark-ink"
                      value={newTripName}
                      onChange={(e) => setNewTripName(e.target.value)}
                    />
                  </div>

                  <div className="space-y-3">
                    <label className="text-xs uppercase font-bold tracking-widest text-teal ml-1">Friends List</label>
                    {friendsInputs.map((friend, index) => (
                      <div key={index} className="flex gap-2">
                        <input 
                          type="text" 
                          placeholder="Name"
                          className="flex-1 bg-vanilla/30 border-2 border-dark-ink/20 rounded-lg px-3 py-2 text-sm outline-none focus:border-teal text-dark-ink"
                          value={friend.name}
                          onChange={(e) => {
                            const newInputs = [...friendsInputs];
                            newInputs[index].name = e.target.value;
                            setFriendsInputs(newInputs);
                          }}
                        />
                        <input 
                          type="tel" 
                          placeholder="Phone (WhatsApp)"
                          className="flex-1 bg-vanilla/30 border-2 border-dark-ink/20 rounded-lg px-3 py-2 text-sm outline-none focus:border-teal text-dark-ink"
                          value={friend.phone}
                          onChange={(e) => {
                            const newInputs = [...friendsInputs];
                            newInputs[index].phone = e.target.value;
                            setFriendsInputs(newInputs);
                          }}
                        />
                      </div>
                    ))}
                    <button 
                      onClick={() => setFriendsInputs([...friendsInputs, { name: "", phone: "" }])}
                      className="text-xs font-bold text-teal flex items-center gap-1 hover:underline"
                    >
                      + Add Another Friend
                    </button>
                  </div>

                  <button 
                    onClick={createTrip}
                    disabled={loading}
                    className="w-full bg-mustard retro-button text-dark-ink py-4 text-lg mt-4 flex items-center justify-center gap-2"
                  >
                    {loading ? "Generating Passport..." : (
                      <>
                        <Plane className="w-5 h-5" />
                        Boarding Now
                      </>
                    )}
                  </button>
                </div>
                
                <div className="pt-6 border-t border-dark-ink/10 flex justify-center">
                   <button onClick={() => setView("admin")} className="text-teal flex items-center gap-1 text-xs font-bold uppercase tracking-wider opacity-60 hover:opacity-100 transition-opacity">
                    <Settings className="w-3 h-3" /> Admin Dashboard
                   </button>
                </div>
              </div>
            </motion.div>
          )}

          {view === "trip" && trip && (
            <motion.div 
              key="trip"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="w-full max-w-4xl grid grid-cols-1 lg:grid-cols-3 gap-8 mt-4"
            >
              {/* Left Column */}
              <div className="lg:col-span-1 space-y-6">
                <div className="retro-card p-6 bg-teal text-white border-dark-ink relative">
                  <div className="retro-tape bg-white/20" />
                  <div className="absolute -top-2 -right-4 rotate-12">
                    <div className="sticker bg-mustard text-dark-ink">ACTIVE TRIP</div>
                  </div>
                  <h2 className="text-2xl mb-1 text-white">{trip.name}</h2>
                  <p className="text-xs font-mono uppercase tracking-widest opacity-80">ID: {trip.id}</p>
                  <div className="mt-6 space-y-4">
                    <div className="flex items-center gap-3">
                      <div className={`w-3 h-3 rounded-full ${trip.status === 'preference_gathering' ? 'bg-mustard animate-pulse' : 'bg-white'}`} />
                      <span className="text-sm">Guest Surveys</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className={`w-3 h-3 rounded-full ${trip.status === 'voting' ? 'bg-mustard animate-pulse' : (trip.status === 'completed' ? 'bg-white' : 'bg-white/20')}`} />
                      <span className={`text-sm ${trip.status === 'preference_gathering' ? 'opacity-40' : ''}`}>AI Consensus & Voting</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className={`w-3 h-3 rounded-full ${trip.status === 'completed' ? 'bg-mustard' : 'bg-white/20'}`} />
                      <span className={`text-sm ${trip.status !== 'completed' ? 'opacity-40' : ''}`}>Final Boarding Call</span>
                    </div>
                  </div>
                </div>

                <MLGroupAnalytics friends={trip.friends} preferences={trip.preferences} />

                <div className="retro-card p-6 space-y-4 bg-white relative">
                   <div className="absolute -bottom-3 -left-3 rotate-12">
                      <div className="sticker flex items-center gap-1"><Users className="w-3 h-3" /> {trip.friends.length}</div>
                   </div>
                   <h3 className="text-xs uppercase font-bold tracking-widest flex items-center gap-2 text-dark-ink">
                     <Users className="w-4 h-4 text-teal" /> Passenger List
                   </h3>
                   <ul className="space-y-3">
                     {trip.friends.map((friend, i) => {
                       const hasSubmitted = trip.preferences.some(p => p.name === friend.name);
                       const hasVoted = trip.votes.some(v => v.name === friend.name);
                       return (
                         <li key={i} className="flex items-center justify-between group">
                           <span className="text-sm font-medium text-dark-ink">{friend.name}</span>
                           <div className="flex items-center gap-2">
                              {hasSubmitted && <span className="text-[10px] bg-teal/10 text-teal px-2 py-0.5 rounded-full font-bold">STAMPED</span>}
                              {!hasSubmitted && <span className="text-[10px] bg-vanilla text-dark-ink/40 px-2 py-0.5 rounded-full font-bold">UNSTAMPED</span>}
                              {hasVoted && <span className="text-[10px] bg-mustard/20 text-mustard px-2 py-0.5 rounded-full font-bold border border-mustard/30">VOTED</span>}
                              {!hasSubmitted && !hasVoted && friend.phone && !isPassenger && (
                                <a 
                                  href={generateWhatsAppLink(friend.phone)} 
                                  target="_blank"
                                  className="text-teal opacity-60 group-hover:opacity-100 transition-opacity"
                                >
                                  <MessageCircle className="w-4 h-4" />
                                </a>
                              )}
                           </div>
                         </li>
                       );
                     })}
                   </ul>
                </div>
              </div>

              {/* Right Column */}
              <div className="lg:col-span-2 space-y-6">
                {trip.status === "generating" && (
                   <div className="retro-card p-12 bg-white flex flex-col items-center justify-center space-y-6">
                      <div className="relative">
                        <div className="w-16 h-16 border-4 border-teal border-t-transparent rounded-full animate-spin" />
                        <Compass className="absolute inset-0 m-auto w-6 h-6 text-teal animate-pulse" />
                      </div>
                      <div className="text-center">
                        <h2 className="text-2xl text-dark-ink font-sans">Calculating Vibe Census...</h2>
                        <p className="text-sm text-dark-ink/60 animate-pulse">Gemini AI is scouting the perfect destinations for your group.</p>
                      </div>
                   </div>
                )}

                {trip.status === "preference_gathering" && (
                  <div className="space-y-6">
                    <div className="retro-card p-8 bg-white">
                      <div className="flex justify-between items-start mb-6">
                        <div>
                          <h2 className="text-2xl text-dark-ink">Preference Survey</h2>
                          <p className="text-sm text-dark-ink/60 italic">What's your vibe?</p>
                        </div>
                        <div className="post-stamp">
                          <MapPin className="text-teal w-6 h-6" />
                        </div>
                      </div>
                      <div className="space-y-4">
                        <select 
                          className="w-full bg-vanilla/10 border-2 border-dark-ink/10 rounded-lg px-4 py-3 outline-none text-dark-ink"
                          value={userName}
                          onChange={(e) => setUserName(e.target.value)}
                        >
                          <option value="">Identify yourself...</option>
                          {trip.friends.map(f => {
                            const alreadySubmitted = trip.preferences.some(p => p.name === f.name);
                            return (
                              <option key={f.name} value={f.name} disabled={alreadySubmitted}>
                                {f.name} {alreadySubmitted ? " (Submitted)" : ""}
                              </option>
                            );
                          })}
                        </select>
                        <textarea 
                          placeholder="Your travel wishes..."
                          className="w-full bg-vanilla/10 border-2 border-dark-ink/10 rounded-lg px-4 py-3 outline-none h-32 text-dark-ink"
                          value={prefText}
                          onChange={(e) => setPrefText(e.target.value)}
                        />
                        <button 
                          onClick={submitPreference}
                          disabled={loading || !userName || !prefText}
                          className="w-full bg-teal text-white retro-button py-4 flex items-center justify-center gap-2"
                        >
                          <Send className="w-5 h-5" /> Submit to AI
                        </button>
                      </div>
                    </div>

                    {trip.preferences.length === trip.friends.length && trip.status === "preference_gathering" && (
                      <div className="flex justify-center">
                        <button onClick={triggerAI} disabled={loading} className="bg-coral text-white retro-button py-6 px-12 text-xl flex flex-col items-center">
                           <div className="flex items-center gap-3"><Cpu /> AI Consensus Failed? Retry</div>
                        </button>
                      </div>
                    )}
                  </div>
                )}

                {trip.status === "voting" && (
                  <div className="space-y-8">
                    <div className="flex justify-between items-end px-2">
                       <div className="space-y-1">
                          <h2 className="text-3xl text-dark-ink">The Round Table</h2>
                          <p className="text-sm opacity-60">Everyone choose their Top 3 rankings.</p>
                       </div>
                       <div className="text-right">
                          <div className="text-[10px] font-bold uppercase tracking-widest text-teal">Participation</div>
                          <div className="text-2xl font-sans">{trip.votes.length} / {trip.friends.length}</div>
                       </div>
                    </div>

                    <div className="grid grid-cols-1 gap-6">
                       {trip.recommendations.map((rec, i) => {
                         const rank = voteRankings.indexOf(i) + 1;
                         return (
                       <div 
                         key={i} 
                         onClick={() => toggleVoteSelection(i)}
                         className={`retro-card cursor-pointer overflow-hidden flex flex-col relative ${rank ? 'border-coral bg-coral/5' : 'bg-white hover:border-teal'}`}
                       >
                          {rank === 1 && <div className="retro-tape bg-coral/30" />}
                          <div className="h-48 w-full overflow-hidden relative border-b-2 border-dark-ink/10">
                             <img 
                               src={`https://source.unsplash.com/featured/800x600?${encodeURIComponent(rec.imageKeyword || rec.destination)}`}
                               alt={rec.destination}
                               className="w-full h-full object-cover transition-transform hover:scale-105 duration-700"
                               referrerPolicy="no-referrer"
                             />
                             <div className="absolute inset-0 bg-dark-ink/5" />
                             <div className={`absolute top-4 right-4 w-10 h-10 flex items-center justify-center rounded-lg border-2 border-dark-ink font-sans transition-all z-10 ${rank ? 'bg-coral text-white scale-110 rotate-12' : 'bg-white shadow-sm'}`}>
                                {rank || i + 1}
                             </div>
                             {rank === 0 && (
                               <div className="absolute bottom-4 left-4 sticker opacity-0 group-hover:opacity-100">CLICK TO VOTE</div>
                             )}
                          </div>
                          <div className="p-6">
                             <h3 className="text-xl text-dark-ink mb-1">{rec.destination}</h3>
                             <p className="text-sm text-dark-ink/70 italic">"{rec.reason}"</p>
                          </div>
                       </div>
                         );
                       })}
                    </div>
                    <div className="retro-card p-8 bg-white border-coral">
                      <select 
                        className="w-full bg-vanilla/10 border-2 border-dark-ink/10 rounded-lg px-4 py-3 outline-none text-dark-ink mb-4"
                        value={userName}
                        onChange={(e) => setUserName(e.target.value)}
                      >
                        <option value="">Identify yourself...</option>
                        {trip.friends.map(f => {
                          const alreadyVoted = trip.votes.some(v => v.name === f.name);
                          return (
                            <option key={f.name} value={f.name} disabled={alreadyVoted}>
                              {f.name} {alreadyVoted ? " (Voted)" : ""}
                            </option>
                          );
                        })}
                      </select>
                      <button onClick={submitVote} disabled={voteRankings.length < 3 || !userName || loading} className="w-full bg-coral text-white retro-button py-3">
                        Submit Rankings
                      </button>
                    </div>
                  </div>
                )}

                {trip.status === "completed" && (
                  <div className="text-center py-12">
                     <h2 className="text-5xl text-dark-ink mb-8">Success!</h2>
                     <div className="retro-card bg-teal text-white overflow-hidden max-w-2xl mx-auto shadow-[8px_8px_0px_0px_rgba(38,70,83,1)]">
                        <div className="h-64 w-full relative">
                           <img 
                             src={`https://source.unsplash.com/featured/1200x800?${encodeURIComponent(trip.winner?.imageKeyword || trip.winner?.destination || 'travel')}`}
                             alt={trip.winner?.destination}
                             className="w-full h-full object-cover opacity-60"
                             referrerPolicy="no-referrer"
                           />
                           <div className="absolute inset-0 bg-gradient-to-t from-teal to-transparent" />
                           <div className="absolute bottom-6 left-0 w-full text-center">
                              <span className="text-[10px] font-mono uppercase tracking-[0.4em] opacity-80 bg-dark-ink/40 px-3 py-1 rounded-full">SELECTED DESTINATION</span>
                           </div>
                        </div>
                        <div className="p-8 sm:p-12 pt-4 space-y-6">
                           <h3 className="text-4xl sm:text-6xl text-white">{trip.winner?.destination}</h3>
                           <p className="text-base sm:text-lg italic text-white/90">"{trip.winner?.reason}"</p>
                           {!isPassenger && (
                             <button onClick={() => setView("home")} className="bg-mustard text-dark-ink retro-button px-10 py-4 mt-6">New Trip</button>
                           )}
                        </div>
                     </div>
                  </div>
                )}
              </div>
            </motion.div>
          )}

          {view === "admin" && (
            <motion.div key="admin" className="w-full max-w-7xl pt-4">
              <div className="min-h-screen pb-12 px-6 bg-[#1a1a1a] text-white font-mono rounded-2xl p-12 border-4 border-white/5 relative overflow-hidden">
                <div className="absolute inset-0 vintage-overlay opacity-20 pointer-events-none" />
                <div className="relative z-10 space-y-8">
                  <div className="flex justify-between items-center mb-12">
                    <h1 className="text-3xl flex items-center gap-3"><Activity className="text-teal" /> SYSTEM_GATEWAY_V1</h1>
                    <button onClick={() => setView("home")} className="px-4 py-2 border border-white/20 rounded-md hover:bg-white/10 text-xs">EXIT_TERMINAL</button>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                    <div className="bg-white/5 p-6 rounded-lg border border-white/10">
                      <div className="text-xs text-teal uppercase mb-2">Request Traffic</div>
                      <div className="text-3xl">{adminStats?.totalRequests || 0}</div>
                    </div>
                    <div className="bg-white/5 p-6 rounded-lg border border-white/10">
                      <div className="text-xs text-coral uppercase mb-2">AI Cost Est</div>
                      <div className="text-3xl">${adminStats?.aiCosts || "0.0000"}</div>
                    </div>
                    <div className="bg-white/5 p-6 rounded-lg border border-white/10">
                      <div className="text-xs text-teal uppercase mb-2">Latency</div>
                      <div className="text-3xl">{adminStats?.avgLatency || "0ms"}</div>
                    </div>
                    <div className="bg-white/5 p-6 rounded-lg border border-white/10">
                      <div className="text-xs text-teal uppercase mb-2">Status</div>
                      <div className="text-3xl">ACTIVE</div>
                    </div>
                  </div>
                  <div className="bg-white/5 border border-white/10 p-6 rounded-lg h-[300px] w-full">
                     <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={adminStats?.history || []}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
                          <XAxis dataKey="time" hide />
                          <YAxis stroke="#ffffff40" />
                          <Tooltip contentStyle={{ backgroundColor: "#1a1a1a" }} />
                          <Area type="monotone" dataKey="ms" stroke="#2A9D8F" fill="#2A9D8F33" />
                        </AreaChart>
                     </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <footer className="fixed bottom-0 w-full py-4 bg-vanilla/90 border-t-2 border-dark-ink/10 flex justify-center gap-12 text-[10px] font-bold uppercase text-dark-ink/40 z-10">
         <span>GPS: 23°N 45°W</span>
         <span>System: Nominal</span>
      </footer>
    </div>
  );
}
