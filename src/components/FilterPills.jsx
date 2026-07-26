import { motion, AnimatePresence } from "framer-motion";

export default function FilterPills({ filters }) {
  const entries = Object.entries(filters || {});
  if (entries.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-2 mb-5">
      <AnimatePresence>
        {entries.map(([key, value]) => (
          <motion.span
            key={key}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="pill-badge bg-gradient-to-r from-primary/10 to-accent/10 text-primary
                       border border-primary/20 font-medium capitalize"
          >
            {key}: {value}
          </motion.span>
        ))}
      </AnimatePresence>
    </div>
  );
}
