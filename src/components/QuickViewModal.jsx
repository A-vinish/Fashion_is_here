import { Dialog } from "@headlessui/react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Heart, Share2, Sparkles } from "lucide-react";
import toast from "react-hot-toast";
import { API_BASE } from "../api";

function RelatedThumb({ item, onSelect }) {
  return (
    <button
      onClick={() => onSelect(item)}
      className="shrink-0 w-16 h-16 rounded-xl overflow-hidden border border-secondary/15
                 hover:border-primary/50 transition-colors"
    >
      <img
        src={`${API_BASE}/image/${item.id}`}
        alt={item.metadata?.category || "related item"}
        className="w-full h-full object-cover"
        onError={(e) => { e.target.style.display = "none"; }}
      />
    </button>
  );
}

export default function QuickViewModal({ item, isFavorite, onToggleFavorite, onClose, relatedItems, onSelectRelated }) {
  const matchPct = item ? Math.max(0, Math.min(100, Math.round((1 - item.distance) * 100))) : 0;
  const meta = item?.metadata || {};

  async function handleShare() {
    const url = `${API_BASE}/image/${item.id}`;
    if (navigator.share) {
      try {
        await navigator.share({ title: `${meta.color} ${meta.category}`, url });
      } catch { /* user cancelled - no-op */ }
    } else {
      await navigator.clipboard.writeText(url);
      toast.success("Image link copied to clipboard");
    }
  }

  return (
    <AnimatePresence>
      {item && (
        <Dialog static open={Boolean(item)} onClose={onClose} className="relative z-[100]">
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-0 bg-ink/50 backdrop-blur-sm"
            aria-hidden="true"
          />
          <div className="fixed inset-0 flex items-center justify-center p-4">
            <Dialog.Panel as={motion.div}
              initial={{ opacity: 0, scale: 0.95, y: 12 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 12 }}
              transition={{ duration: 0.2 }}
              className="relative w-full max-w-3xl max-h-[88vh] overflow-y-auto bg-white
                         rounded-3xl shadow-card-hover grid grid-cols-1 sm:grid-cols-2"
            >
              <button
                onClick={onClose}
                aria-label="Close quick view"
                className="absolute top-4 right-4 z-10 w-9 h-9 rounded-full bg-white shadow-card
                           flex items-center justify-center text-ink hover:rotate-90 transition-transform"
              >
                <X size={16} />
              </button>

              <div className="relative bg-bg">
                <img
                  src={`${API_BASE}/image/${item.id}`}
                  alt={meta.category || "fashion item"}
                  className="w-full h-full object-cover min-h-[280px]"
                />
                <span className="absolute top-4 left-4 pill-badge bg-gradient-to-r from-primary to-accent text-white shadow-glow">
                  🔥 {matchPct}% Match
                </span>
              </div>

              <div className="p-7">
                <Dialog.Title className="font-display font-semibold text-2xl text-ink capitalize">
                  {meta.color} {meta.category}
                </Dialog.Title>
                <p className="text-muted italic mt-1">For {meta.occasion || "any"} occasions</p>

                <div className="flex items-center justify-between py-5 mt-4 border-y border-secondary/10">
                  <span className="font-display font-semibold text-2xl text-primary">₹{meta.price ?? "?"}</span>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={handleShare}
                      aria-label="Share this item"
                      className="w-10 h-10 rounded-full border border-secondary/20 flex items-center justify-center text-ink hover:border-secondary/50"
                    >
                      <Share2 size={16} />
                    </button>
                    <button
                      onClick={() => onToggleFavorite(item.id)}
                      className={`flex items-center gap-2 rounded-full px-4 py-2.5 text-sm font-medium border
                                  ${isFavorite ? "bg-accent text-white border-accent" : "border-secondary/30 text-ink"}`}
                    >
                      <Heart size={15} fill={isFavorite ? "currentColor" : "none"} />
                      {isFavorite ? "Saved" : "Save"}
                    </button>
                  </div>
                </div>

                {/* AI explanation - grounded in the REAL similarity score, no fabricated claims */}
                <div className="mt-5 flex gap-2.5 rounded-2xl bg-primary/5 p-4">
                  <Sparkles size={16} className="text-primary shrink-0 mt-0.5" />
                  <p className="text-sm text-ink/80 leading-relaxed">
                    CLIP's visual embedding places this item at <strong>{matchPct}% similarity</strong> to
                    your search — based on shared color, silhouette, and visual style detected by the model.
                  </p>
                </div>

                {relatedItems.length > 0 && (
                  <div className="mt-6">
                    <p className="text-xs uppercase tracking-widest text-muted mb-2.5">You may also like</p>
                    <div className="flex gap-2.5 overflow-x-auto pb-1">
                      {relatedItems.map((r) => (
                        <RelatedThumb key={r.id} item={r} onSelect={onSelectRelated} />
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </Dialog.Panel>
          </div>
        </Dialog>
      )}
    </AnimatePresence>
  );
}
