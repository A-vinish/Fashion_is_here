import { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ShoppingBag } from "lucide-react";
import ProductCard from "./ProductCard.jsx";
import FiltersBar from "./FiltersBar.jsx";
import QuickViewModal from "./QuickViewModal.jsx";

function SkeletonCard({ index }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: index * 0.04 }}
      className="bg-white rounded-3xl overflow-hidden shadow-card"
    >
      <div className="h-52 bg-gradient-to-r from-secondary/10 via-secondary/20 to-secondary/10
                       bg-[length:200%_100%] animate-shimmer" />
      <div className="p-4 space-y-2">
        <div className="h-3.5 w-2/3 rounded bg-secondary/10 animate-pulse" />
        <div className="h-3 w-1/2 rounded bg-secondary/10 animate-pulse" />
      </div>
    </motion.div>
  );
}

export default function ResultsGrid({ results, isLoading }) {
  const [sortBy, setSortBy] = useState("match");
  const [activeCategory, setActiveCategory] = useState(null);
  const [activeOccasion, setActiveOccasion] = useState(null);
  const [favorites, setFavorites] = useState(() => new Set());
  const [quickViewItem, setQuickViewItem] = useState(null);

  function toggleFavorite(id) {
    setFavorites((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  const categories = useMemo(
    () => [...new Set((results || []).map((r) => r.metadata?.category).filter(Boolean))],
    [results]
  );
  const occasions = useMemo(
    () => [...new Set((results || []).map((r) => r.metadata?.occasion).filter(Boolean))],
    [results]
  );

  const filteredSorted = useMemo(() => {
    if (!results) return [];
    let list = [...results];
    if (activeCategory) list = list.filter((r) => r.metadata?.category === activeCategory);
    if (activeOccasion) list = list.filter((r) => r.metadata?.occasion === activeOccasion);

    if (sortBy === "price-asc") list.sort((a, b) => (a.metadata?.price ?? 0) - (b.metadata?.price ?? 0));
    else if (sortBy === "price-desc") list.sort((a, b) => (b.metadata?.price ?? 0) - (a.metadata?.price ?? 0));
    else list.sort((a, b) => a.distance - b.distance);

    return list;
  }, [results, sortBy, activeCategory, activeOccasion]);

  if (isLoading) {
    return (
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-5">
        {Array.from({ length: 8 }).map((_, i) => <SkeletonCard key={i} index={i} />)}
      </div>
    );
  }

  if (!results || results.length === 0) {
    return (
      <div className="flex flex-col items-center gap-3 text-center py-16 px-6 rounded-3xl
                       border-2 border-dashed border-secondary/20 bg-white/50">
        <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-primary/10 to-accent/10
                         flex items-center justify-center text-primary">
          <ShoppingBag size={26} strokeWidth={1.4} />
        </div>
        <p className="text-muted max-w-xs">
          Nothing on the rail yet — upload an image or describe what you're looking for.
        </p>
      </div>
    );
  }

  const hasActiveFilters = Boolean(activeCategory || activeOccasion);

  return (
    <>
      <FiltersBar
        sortBy={sortBy}
        onSortChange={setSortBy}
        categories={categories}
        activeCategory={activeCategory}
        onCategoryToggle={(c) => setActiveCategory((cur) => (cur === c ? null : c))}
        occasions={occasions}
        activeOccasion={activeOccasion}
        onOccasionToggle={(o) => setActiveOccasion((cur) => (cur === o ? null : o))}
        hasActiveFilters={hasActiveFilters}
        onClearFilters={() => { setActiveCategory(null); setActiveOccasion(null); }}
        resultCount={filteredSorted.length}
      />

      <motion.div layout className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-5">
        <AnimatePresence>
          {filteredSorted.map((item, i) => (
            <ProductCard
              key={item.id}
              item={item}
              index={i}
              isFavorite={favorites.has(item.id)}
              onToggleFavorite={toggleFavorite}
              onQuickView={setQuickViewItem}
            />
          ))}
        </AnimatePresence>
      </motion.div>

      <QuickViewModal
        item={quickViewItem}
        isFavorite={quickViewItem ? favorites.has(quickViewItem.id) : false}
        onToggleFavorite={toggleFavorite}
        onClose={() => setQuickViewItem(null)}
        relatedItems={
          quickViewItem
            ? filteredSorted.filter((r) => r.id !== quickViewItem.id).slice(0, 4)
            : []
        }
        onSelectRelated={setQuickViewItem}
      />
    </>
  );
}
