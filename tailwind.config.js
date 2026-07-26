/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        primary: "#6D28D9",
        secondary: "#9333EA",
        accent: "#EC4899",
        bg: "#F5F4FA",
        ink: "#1F2937",
        muted: "#6B7280",
        success: "#22C55E",
        danger: "#EF4444",
      },
      fontFamily: {
        display: ["'Playfair Display'", "serif"],
        body: ["Inter", "sans-serif"],
        button: ["Poppins", "sans-serif"],
      },
      boxShadow: {
        glow: "0 8px 30px rgba(109,40,217,0.25)",
        card: "0 4px 24px rgba(31,41,55,0.06)",
        "card-hover": "0 12px 40px rgba(109,40,217,0.14)",
      },
      backdropBlur: {
        xs: "2px",
      },
      keyframes: {
        blob: {
          "0%, 100%": { transform: "translate(0,0) scale(1)" },
          "33%": { transform: "translate(30px,-40px) scale(1.08)" },
          "66%": { transform: "translate(-20px,20px) scale(0.95)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "200% 0" },
          "100%": { backgroundPosition: "-200% 0" },
        },
      },
      animation: {
        blob: "blob 14s ease-in-out infinite",
        shimmer: "shimmer 1.8s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
