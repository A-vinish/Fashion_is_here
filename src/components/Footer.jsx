import { Github, Linkedin, Mail, Sparkles } from "lucide-react";

export default function Footer() {
  return (
    <footer className="border-t border-secondary/10 mt-16 pt-10 pb-8">
      <div className="flex flex-col sm:flex-row items-center justify-between gap-6">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center text-white">
            <Sparkles size={15} />
          </div>
          <span className="font-display font-semibold text-ink">StyleGPT</span>
        </div>

        <div className="flex items-center gap-5 text-sm text-muted">
          <a href="#" className="hover:text-primary transition-colors">Privacy</a>
          <a href="#" className="hover:text-primary transition-colors">Terms</a>
          <a href="#" className="hover:text-primary transition-colors">Contact</a>
        </div>

        <div className="flex items-center gap-3">
          <a href="#" aria-label="GitHub" className="w-9 h-9 rounded-full border border-secondary/20 flex items-center justify-center text-ink hover:border-primary hover:text-primary transition-colors">
            <Github size={16} />
          </a>
          <a href="#" aria-label="LinkedIn" className="w-9 h-9 rounded-full border border-secondary/20 flex items-center justify-center text-ink hover:border-primary hover:text-primary transition-colors">
            <Linkedin size={16} />
          </a>
          <a href="#" aria-label="Email" className="w-9 h-9 rounded-full border border-secondary/20 flex items-center justify-center text-ink hover:border-primary hover:text-primary transition-colors">
            <Mail size={16} />
          </a>
        </div>
      </div>
      <p className="text-center text-xs text-muted mt-8">
        Built with CLIP, ChromaDB, and LangChain — an AI fashion search demo.
      </p>
    </footer>
  );
}
