import { motion } from "framer-motion";
import { Palette, Tag, Sparkle, Target } from "lucide-react";

/**
 * NOTE on honesty: the backend's CLIP pipeline never classifies the
 * *uploaded* query photo itself (only catalog items get auto-tagged
 * during seeding, see auto_tag.py). So this panel does NOT claim to
 * detect "material / pattern / season / confidence" for your photo -
 * those fields don't exist anywhere in the real pipeline, and making
 * them up would be fabricated UI. Instead, this shows the REAL,
 * backend-computed metadata and similarity score of your closest match.
 */
export default function AnalysisPanel({ topMatch }) {
  if (!topMatch) return null;

  const matchPct = Math.max(0, Math.min(100, Math.round((1 - topMatch.distance) * 100)));
  const meta = topMatch.metadata || {};

  const fields = [
    { icon: Tag, label: "Category", value: meta.category },
    { icon: Palette, label: "Color", value: meta.color },
    { icon: Sparkle, label: "Occasion", value: meta.occasion },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card p-6 mb-6"
    >
      <div className="flex items-center justify-between mb-5">
        <h3 className="font-display font-semibold text-lg text-ink">Closest Match Analysis</h3>
        <span className="pill-badge bg-primary/10 text-primary">
          <Target size={12} /> Real-time
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 items-center">
        {fields.map(({ icon: Icon, label, value }) => (
          <div key={label} className="flex flex-col gap-1.5">
            <span className="flex items-center gap-1.5 text-xs uppercase tracking-wide text-muted">
              <Icon size={12} /> {label}
            </span>
            <span className="font-medium text-ink capitalize">{value || "—"}</span>
          </div>
        ))}

        {/* Animated confidence ring - built from the REAL cosine-similarity score */}
        <div className="flex items-center gap-3 justify-self-start sm:justify-self-end">
          <div className="relative w-16 h-16">
            <svg viewBox="0 0 64 64" className="w-16 h-16 -rotate-90">
              <circle cx="32" cy="32" r="27" fill="none" stroke="#E5E7EB" strokeWidth="6" />
              <motion.circle
                cx="32" cy="32" r="27" fill="none" strokeWidth="6" strokeLinecap="round"
                stroke="url(#matchGradient)"
                strokeDasharray={2 * Math.PI * 27}
                initial={{ strokeDashoffset: 2 * Math.PI * 27 }}
                animate={{ strokeDashoffset: 2 * Math.PI * 27 * (1 - matchPct / 100) }}
                transition={{ duration: 0.8, ease: "easeOut" }}
              />
              <defs>
                <linearGradient id="matchGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#6D28D9" />
                  <stop offset="100%" stopColor="#EC4899" />
                </linearGradient>
              </defs>
            </svg>
            <span className="absolute inset-0 flex items-center justify-center text-sm font-semibold text-ink">
              {matchPct}%
            </span>
          </div>
          <div className="text-xs text-muted leading-tight">
            Visual<br />Similarity
          </div>
        </div>
      </div>
    </motion.div>
  );
}
