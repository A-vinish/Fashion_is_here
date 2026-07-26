import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Sparkles } from "lucide-react";

// These map to what's ACTUALLY happening in the pipeline (embeddings.py ->
// vectorstore.py -> filter_parser.py), not made-up filler text.
const STAGE_MESSAGES = [
  "Reading your image with CLIP...",
  "Matching visual embeddings...",
  "Searching the catalog...",
  "Applying your style filters...",
];

export default function LoadingExperience() {
  const [stageIndex, setStageIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setStageIndex((i) => (i + 1) % STAGE_MESSAGES.length);
    }, 1100);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-col items-center gap-4 py-10">
      <div className="relative w-12 h-12">
        <div className="absolute inset-0 rounded-full border-2 border-secondary/20" />
        <motion.div
          className="absolute inset-0 rounded-full border-2 border-t-primary border-r-accent border-b-transparent border-l-transparent"
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
        />
        <Sparkles size={16} className="absolute inset-0 m-auto text-primary" />
      </div>

      {/* Indeterminate progress bar - honest: we don't know real % complete,
          so this animates as a moving stripe, not a fabricated percentage. */}
      <div className="w-56 h-1.5 rounded-full bg-secondary/15 overflow-hidden">
        <motion.div
          className="h-full w-1/3 rounded-full bg-gradient-to-r from-primary to-accent"
          animate={{ x: ["-100%", "220%"] }}
          transition={{ duration: 1.3, repeat: Infinity, ease: "easeInOut" }}
        />
      </div>

      <AnimatePresence mode="wait">
        <motion.p
          key={stageIndex}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
          transition={{ duration: 0.25 }}
          className="text-sm text-muted font-medium"
        >
          {STAGE_MESSAGES[stageIndex]}
        </motion.p>
      </AnimatePresence>
    </div>
  );
}
