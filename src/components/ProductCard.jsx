import { motion } from "framer-motion";
import { Heart, Eye } from "lucide-react";
import { API_BASE } from "../api";

/**
 * NOTE on honesty: the spec asked for Brand / Rating / Discount chips too,
 * but none of those exist anywhere in our data (auto_tag.py only produces
 * color/category/occasion/price - see backend). Showing a star rating or a
 * "20% off" badge with no real number behind it would be fabricated, so
 * this card only surfaces fields that are genuinely computed by the pipeline.
 */
export default function ProductCard({ item, index, isFavorite, onToggleFavorite, onQuickView }) {
  const imgSrc = `${API_BASE}/image/${item.id}`;
  const matchPct = Math.max(0, Math.min(100, Math.round((1 - item.distance) * 100)));
  const meta = item.metadata || {};

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay: Math.min(index, 10) * 0.04 }}
      whileHover={{ y: -6 }}
      className="group bg-white rounded-3xl overflow-hidden shadow-card hover:shadow-card-hover transition-shadow duration-300"
    >
      <div className="relative overflow-hidden cursor-pointer" onClick={() => onQuickView(item)}>
        <img
          src={imgSrc}
          alt={meta.category || "fashion item"}
          onError={(e) => { e.target.style.display = "none"; }}
          className="w-full h-52 object-cover transition-transform duration-500 group-hover:scale-110"
        />

        <span className="absolute top-3 left-3 pill-badge bg-gradient-to-r from-primary to-accent text-white shadow-glow">
          🔥 {matchPct}% Match
        </span>

        <button
          onClick={(e) => { e.stopPropagation(); onToggleFavorite(item.id); }}
          aria-label={isFavorite ? "Remove from wishlist" : "Add to wishlist"}
          className={`absolute top-3 right-3 w-8 h-8 rounded-full flex items-center justify-center
                      transition-colors duration-150 shadow-card
                      ${isFavorite ? "bg-accent text-white" : "bg-white/90 text-ink hover:bg-white"}`}
        >
          <Heart size={15} fill={isFavorite ? "currentColor" : "none"} />
        </button>

        <div className="absolute inset-0 bg-gradient-to-t from-ink/60 via-transparent to-transparent
                         opacity-0 group-hover:opacity-100 transition-opacity duration-200
                         flex items-end justify-center pb-4">
          <span className="pill-badge bg-white text-ink text-xs font-medium shadow-card">
            <Eye size={13} /> Quick View
          </span>
        </div>
      </div>

      <div className="p-4">
        <h4 className="font-display font-semibold text-ink capitalize truncate">{meta.category || "item"}</h4>
        <p className="text-sm text-muted capitalize mt-0.5">{meta.color} · {meta.occasion}</p>
        <div className="flex items-center justify-between mt-3">
          <span className="font-semibold text-ink">₹{meta.price ?? "?"}</span>
        </div>
      </div>
    </motion.div>
  );
}
