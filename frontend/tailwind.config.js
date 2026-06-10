/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        outfit: ['Outfit', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      colors: {
        darkBg: '#090a0f',
        darkCard: 'rgba(15, 18, 28, 0.65)',
        accentGreen: '#00e676',
        accentRed: '#ff1744',
        accentOrange: '#ff9100',
        accentBlue: '#2979ff',
        textPrimary: '#ffffff',
        textSecondary: '#8a99ad',
      },
      boxShadow: {
        glass: '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
        neonGreen: '0 0 15px rgba(0, 230, 118, 0.4)',
        neonRed: '0 0 15px rgba(255, 23, 68, 0.4)',
      }
    },
  },
  plugins: [],
}
