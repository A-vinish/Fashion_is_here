import { motion } from "framer-motion";
import { Upload, MessageSquare } from "lucide-react";

export default function Hero({ onUploadClick, onDescribeClick }) {
  return (
    <section className="relative overflow-hidden pt-20 pb-16 px-6 text-center">
      {/* Floating blurred gradient blobs - purely decorative background */}
      <div className="blob w-72 h-72 bg-primary -top-10 -left-16 animate-blob" aria-hidden="true" />
      <div
        className="blob w-80 h-80 bg-accent top-10 -right-20 animate-blob"
        style={{ animationDelay: "3s" }}
        aria-hidden="true"
      />
      <div
        className="blob w-64 h-64 bg-secondary bottom-0 left-1/3 animate-blob"
        style={{ animationDelay: "6s" }}
        aria-hidden="true"
      />

      <div className="relative z-10 max-w-2xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="pill-badge bg-white/70 backdrop-blur-sm border border-white/60 text-primary mb-6 shadow-card"
        >
          <span className="w-1.5 h-1.5 rounded-full bg-accent" />
          Powered by CLIP Vision + LangChain
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="font-display font-semibold text-5xl sm:text-6xl leading-[1.08] gradient-text"
        >
          AI Personal
          <br />
          Fashion Stylist
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="mt-5 text-lg text-muted max-w-lg mx-auto"
        >
          Find visually similar outfits using AI computer vision — upload a look,
          or just describe what you have in mind.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="mt-9 flex flex-wrap items-center justify-center gap-3"
        >
          <button onClick={onUploadClick} className="gradient-btn inline-flex items-center gap-2">
            <Upload size={16} /> Upload Image
          </button>
          <button
            onClick={onDescribeClick}
            className="inline-flex items-center gap-2 rounded-full px-6 py-3 font-button font-medium
                       text-primary bg-white/70 backdrop-blur-sm border border-white/60 shadow-card
                       transition-transform duration-200 hover:scale-[1.03]"
          >
            <MessageSquare size={16} /> Describe Outfit
          </button>
        </motion.div>
      </div>
    </section>
  );
}
