import { Listbox } from "@headlessui/react";
import { ChevronDown, SlidersHorizontal, X } from "lucide-react";

const SORT_OPTIONS = [
  { value: "match", label: "Best Match" },
  { value: "price-asc", label: "Price: Low to High" },
  { value: "price-desc", label: "Price: High to Low" },
];

function SortDropdown({ value, onChange }) {
  const current = SORT_OPTIONS.find((o) => o.value === value) || SORT_OPTIONS[0];
  return (
    <Listbox value={value} onChange={onChange}>
      <div className="relative">
        <Listbox.Button className="flex items-center gap-2 rounded-full bg-white border border-secondary/20
                                    px-4 py-2 text-sm font-medium text-ink shadow-card
                                    focus:outline-none focus:ring-2 focus:ring-primary/40">
          <SlidersHorizontal size={14} className="text-secondary" />
          {current.label}
          <ChevronDown size={14} className="text-muted" />
        </Listbox.Button>
        <Listbox.Options className="absolute z-20 mt-2 w-52 rounded-2xl bg-white shadow-card-hover
                                     border border-secondary/10 py-1 focus:outline-none">
          {SORT_OPTIONS.map((opt) => (
            <Listbox.Option
              key={opt.value}
              value={opt.value}
              className={({ active, selected }) =>
                `px-4 py-2 text-sm cursor-pointer ${active ? "bg-primary/5" : ""} ${selected ? "text-primary font-medium" : "text-ink"}`
              }
            >
              {opt.label}
            </Listbox.Option>
          ))}
        </Listbox.Options>
      </div>
    </Listbox>
  );
}

function ChipGroup({ label, options, active, onToggle }) {
  if (options.length === 0) return null;
  return (
    <div className="flex items-center gap-2 flex-wrap">
      <span className="text-xs uppercase tracking-wide text-muted mr-1">{label}</span>
      {options.map((opt) => (
        <button
          key={opt}
          onClick={() => onToggle(opt)}
          className={`px-3 py-1.5 rounded-full text-xs font-medium capitalize transition-colors duration-150
                      ${active === opt
                        ? "bg-gradient-to-r from-primary to-secondary text-white"
                        : "bg-white text-ink border border-secondary/15 hover:border-secondary/40"}`}
        >
          {opt}
        </button>
      ))}
    </div>
  );
}

export default function FiltersBar({
  sortBy, onSortChange,
  categories, activeCategory, onCategoryToggle,
  occasions, activeOccasion, onOccasionToggle,
  hasActiveFilters, onClearFilters,
  resultCount,
}) {
  return (
    <div className="flex flex-col gap-4 mb-6">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <span className="text-xs uppercase tracking-widest text-muted">
          {resultCount} item{resultCount === 1 ? "" : "s"}
        </span>
        <div className="flex items-center gap-2">
          {hasActiveFilters && (
            <button
              onClick={onClearFilters}
              className="flex items-center gap-1 text-xs text-muted hover:text-accent"
            >
              <X size={12} /> Clear filters
            </button>
          )}
          <SortDropdown value={sortBy} onChange={onSortChange} />
        </div>
      </div>

      <div className="flex flex-wrap gap-4">
        <ChipGroup label="Category" options={categories} active={activeCategory} onToggle={onCategoryToggle} />
        <ChipGroup label="Occasion" options={occasions} active={activeOccasion} onToggle={onOccasionToggle} />
      </div>
    </div>
  );
}
