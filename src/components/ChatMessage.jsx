import { motion } from "framer-motion";
import { Sparkles } from "lucide-react";

export default function ChatMessage({ text, sender }) {
  const isBot = sender === "bot";
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className={`flex items-end gap-2 mb-3 ${isBot ? "" : "justify-end"}`}
    >
      {isBot && (
        <div className="w-6 h-6 rounded-full bg-gradient-to-br from-primary to-accent
                         flex items-center justify-center text-white shrink-0">
          <Sparkles size={12} />
        </div>
      )}
      <div
        className={`max-w-[75%] px-4 py-2.5 text-sm leading-relaxed rounded-2xl
                    ${isBot
                      ? "bg-white text-ink rounded-bl-sm shadow-card"
                      : "bg-gradient-to-r from-primary to-secondary text-white rounded-br-sm"}`}
      >
        {text}
      </div>
    </motion.div>
  );
}
