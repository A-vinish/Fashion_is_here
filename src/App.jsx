import { useState, useRef, useEffect } from "react";
import toast from "react-hot-toast";
import { RotateCcw } from "lucide-react";
import Hero from "./components/Hero.jsx";
import UploadDropzone from "./components/UploadDropzone.jsx";
import ChatMessage from "./components/ChatMessage.jsx";
import FilterPills from "./components/FilterPills.jsx";
import ResultsGrid from "./components/ResultsGrid.jsx";
import AnalysisPanel from "./components/AnalysisPanel.jsx";
import LoadingExperience from "./components/LoadingExperience.jsx";
import StatsStrip from "./components/StatsStrip.jsx";
import FeatureSection from "./components/FeatureSection.jsx";
import Testimonials from "./components/Testimonials.jsx";
import Footer from "./components/Footer.jsx";
import { searchByImage, sendChatMessage, resetSession, getHealth } from "./api.js";

const sessionId = "session_" + Math.random().toString(36).slice(2);

export default function App() {
  const [messages, setMessages] = useState([
    { sender: "bot", text: "Hello — upload an image to start, or tell me what you have in mind." },
  ]);
  const [filters, setFilters] = useState({});
  const [results, setResults] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [isBusy, setIsBusy] = useState(false);
  const [catalogSize, setCatalogSize] = useState(null);
  const [lastSearchTimeMs, setLastSearchTimeMs] = useState(null);

  const chatBoxRef = useRef(null);
  const uploadSectionRef = useRef(null);
  const chatSectionRef = useRef(null);

  useEffect(() => {
    getHealth()
      .then((data) => setCatalogSize(data.items_in_catalog))
      .catch(() => {}); // stats strip just shows "—" if this fails, non-critical
  }, []);

  useEffect(() => {
    if (chatBoxRef.current) {
      chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
    }
  }, [messages]);

  function addMessage(text, sender) {
    setMessages((prev) => [...prev, { text, sender }]);
  }

  async function handleFileSelected(file) {
    if (!file) return; // cleared, nothing to search
    addMessage("Searching by uploaded image...", "user");
    setIsBusy(true);
    const startedAt = performance.now();
    try {
      const data = await searchByImage(file, sessionId);
      const elapsed = performance.now() - startedAt;
      setLastSearchTimeMs(elapsed);
      addMessage(`Found ${data.results.length} similar items.`, "bot");
      setResults(data.results);
    } catch (err) {
      toast.error(err.message);
    } finally {
      setIsBusy(false);
    }
  }

  async function handleSendChat() {
    const message = chatInput.trim();
    if (!message) return;

    addMessage(message, "user");
    setChatInput("");
    setIsBusy(true);
    const startedAt = performance.now();
    try {
      const data = await sendChatMessage(message, sessionId);
      const elapsed = performance.now() - startedAt;
      setLastSearchTimeMs(elapsed);
      addMessage(data.reply, "bot");
      setFilters(data.applied_filters || {});
      setResults(data.results || []);
    } catch (err) {
      toast.error(err.message);
    } finally {
      setIsBusy(false);
    }
  }

  function handleInputKeyDown(e) {
    if (e.key === "Enter") handleSendChat();
  }

  function handleNewSearch() {
    setMessages([{ sender: "bot", text: "Starting fresh — upload an image or tell me what you're looking for." }]);
    setFilters({});
    setResults([]);
    setChatInput("");
    setLastSearchTimeMs(null);
    resetSession(sessionId).catch(() => {});
    toast.success("Started a new search");
  }

  function scrollTo(ref) {
    ref.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  const showLoading = isBusy && results.length === 0;
  const topMatch = results.length > 0 ? results[0] : null;

  return (
    <div className="min-h-screen">
      <Hero
        onUploadClick={() => scrollTo(uploadSectionRef)}
        onDescribeClick={() => scrollTo(chatSectionRef)}
      />

      <div className="max-w-5xl mx-auto px-6">
        <div ref={uploadSectionRef} className="scroll-mt-8 mb-10">
          <UploadDropzone onFileSelected={handleFileSelected} isBusy={isBusy} />
        </div>

        <StatsStrip catalogSize={catalogSize} searchTimeMs={lastSearchTimeMs} matchCount={results.length || null} />

        {showLoading && <LoadingExperience />}
        {!showLoading && topMatch && <AnalysisPanel topMatch={topMatch} />}

        <div ref={chatSectionRef} className="scroll-mt-8 mb-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-display font-semibold text-lg text-ink">Chat with your Stylist</h2>
            <button
              onClick={handleNewSearch}
              className="flex items-center gap-1.5 text-xs text-muted hover:text-primary transition-colors"
            >
              <RotateCcw size={12} /> New Search
            </button>
          </div>

          <FilterPills filters={filters} />

          <div ref={chatBoxRef} className="glass-card p-5 max-h-96 overflow-y-auto mb-4">
            {messages.map((m, i) => <ChatMessage key={i} text={m.text} sender={m.sender} />)}
          </div>

          <div className="flex gap-2 mb-10">
            <input
              type="text"
              placeholder="Refine your search… e.g. 'but in blue, for a wedding'"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={handleInputKeyDown}
              disabled={isBusy}
              className="flex-1 rounded-full px-5 py-3 bg-white border border-secondary/20 text-sm
                         focus:outline-none focus:ring-2 focus:ring-primary/40 disabled:opacity-60"
            />
            <button onClick={handleSendChat} disabled={isBusy} className="gradient-btn">
              {isBusy ? "..." : "Send"}
            </button>
          </div>
        </div>

        <ResultsGrid results={results} isLoading={showLoading} />

        <FeatureSection />
        <Testimonials />
      </div>

      <div className="max-w-5xl mx-auto px-6">
        <Footer />
      </div>
    </div>
  );
}
