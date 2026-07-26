import { motion } from "framer-motion";
import { Eye, Search, Sparkles, Heart, MessageCircle, Layers } from "lucide-react";

const FEATURES = [
  { icon: Eye, title: "AI Vision", desc: "CLIP reads the color, shape, and style of any photo you upload." },
  { icon: Search, title: "Visual Search", desc: "Finds catalog items that look similar — not just ones with matching tags." },
  { icon: MessageCircle, title: "Conversational Refinement", desc: "Say \"but in blue, for a wedding\" and the AI narrows results live." },
  { icon: Layers, title: "Hybrid Matching", desc: "Combines visual similarity with real filters like price and occasion." },
  { icon: Heart, title: "Wishlist", desc: "Save items you like as you browse through the rail." },
  { icon: Sparkles, title: "Personal Styling", desc: "A LangChain-powered assistant that remembers your preferences mid-chat." },
];

export default function FeatureSection() {
  return (
    <section className="py-16">
      <div className="text-center mb-12">
        <span className="text-xs uppercase tracking-widest text-primary font-medium">How it works</span>
        <h2 className="font-display font-semibold text-3xl mt-2 text-ink">Built on real AI, not gimmicks</h2>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
        {FEATURES.map(({ icon: Icon, title, desc }, i) => (
          <motion.div
            key={title}
            initial={{ opacity: 0, y: 14 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-40px" }}
            transition={{ delay: i * 0.06 }}
            className="glass-card p-6 hover:shadow-card-hover transition-shadow duration-300"
          >
            <div className="w-11 h-11 rounded-2xl bg-gradient-to-br from-primary to-secondary
                             flex items-center justify-center text-white mb-4">
              <Icon size={19} />
            </div>
            <h3 className="font-display font-semibold text-ink mb-1.5">{title}</h3>
            <p className="text-sm text-muted leading-relaxed">{desc}</p>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
