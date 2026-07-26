import { motion } from "framer-motion";

/**
 * NOTE on honesty: the spec asked for numbers like "50K+ Products" and
 * "100K+ Successful Recommendations" - those would be fabricated for a
 * real-portfolio project with an actual (much smaller, self-seeded) catalog.
 * Instead these three numbers are all genuinely measured: catalogSize comes
 * straight from the backend's /health endpoint, and searchTimeMs / matchCount
 * are measured live from the most recent real search.
 */
export default function StatsStrip({ catalogSize, searchTimeMs, matchCount }) {
  const stats = [
    { value: catalogSize != null ? catalogSize.toLocaleString() : "—", label: "Items in Catalog" },
    { value: searchTimeMs != null ? `${(searchTimeMs / 1000).toFixed(2)}s` : "—", label: "Last Search Time" },
    { value: matchCount != null ? matchCount : "—", label: "Matches Found" },
  ];

  return (
    <div className="grid grid-cols-3 gap-4 mb-10">
      {stats.map((s, i) => (
        <motion.div
          key={s.label}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.08 }}
          className="glass-card text-center py-5"
        >
          <div className="font-display font-semibold text-2xl gradient-text">{s.value}</div>
          <div className="text-xs text-muted mt-1 uppercase tracking-wide">{s.label}</div>
        </motion.div>
      ))}
    </div>
  );
}
