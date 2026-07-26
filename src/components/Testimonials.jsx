import { motion } from "framer-motion";

/**
 * NOTE on honesty: this is a portfolio/demo project with no real users yet,
 * so these are clearly-labeled SAMPLE quotes, not real customer testimonials.
 * Presenting invented names/quotes as real feedback would be misleading -
 * swap these for genuine feedback once real people have tried the app.
 */
const SAMPLE_TESTIMONIALS = [
  { name: "Sample User A", role: "Early tester", quote: "Uploading a photo and getting visually similar pieces back felt genuinely useful, not gimmicky." },
  { name: "Sample User B", role: "Early tester", quote: "Being able to say 'but in blue for a wedding' and have it actually understand was the standout moment." },
  { name: "Sample User C", role: "Early tester", quote: "The hybrid search — combining what it looks like with real filters — is exactly what normal keyword search is missing." },
];

export default function Testimonials() {
  return (
    <section className="py-16">
      <div className="text-center mb-3">
        <span className="text-xs uppercase tracking-widest text-primary font-medium">Early feedback</span>
        <h2 className="font-display font-semibold text-3xl mt-2 text-ink">What early testers say</h2>
      </div>
      <p className="text-center text-xs text-muted mb-10">
        (Sample quotes from early testing — not yet real production reviews)
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-5">
        {SAMPLE_TESTIMONIALS.map((t, i) => (
          <motion.div
            key={t.name}
            initial={{ opacity: 0, y: 14 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-40px" }}
            transition={{ delay: i * 0.08 }}
            className="bg-white rounded-3xl p-6 shadow-card"
          >
            <p className="text-sm text-ink/80 leading-relaxed mb-5">"{t.quote}"</p>
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-full bg-gradient-to-br from-primary to-accent
                               flex items-center justify-center text-white text-xs font-semibold">
                {t.name.split(" ").map((w) => w[0]).join("")}
              </div>
              <div>
                <div className="text-sm font-medium text-ink">{t.name}</div>
                <div className="text-xs text-muted">{t.role}</div>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
