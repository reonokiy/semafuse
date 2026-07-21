import { defineConfig, presetWind3 } from "unocss";

export default defineConfig({
  presets: [presetWind3()],
  theme: {
    breakpoints: {
      sm: "40rem",
      mobile: "45rem",
      md: "48rem",
      tablet: "62.5rem",
      lg: "64rem",
      xl: "80rem",
      "2xl": "96rem",
    },
    colors: {
      paper: "#f1eee6",
      "paper-deep": "#e8e3d7",
      ink: "#17221d",
      "ink-soft": "#3e4a44",
      faded: "#8b9189",
      jade: "#b9d6c1",
      "jade-dark": "#648b72",
      seal: "#bd4a35",
      night: "#111a16",
    },
    fontFamily: {
      body: '"Inter Variable", Inter, sans-serif',
      mono: '"IBM Plex Mono", monospace',
      script: '"Cormorant Garamond", Georgia, serif',
    },
    animation: {
      keyframes: {
        "signal-in": "{to{opacity:1}}",
        "draw-thread": "{to{stroke-dashoffset:0}}",
        "agent-in": "{from{opacity:0;transform:translateY(0.5rem)}to{opacity:1;transform:translateY(0)}}",
        "core-in": "{from{opacity:0;transform:translate(-50%,-50%) scale(.8)}to{opacity:1;transform:translate(-50%,-50%) scale(1)}}",
        "scroll-line": "{from{transform:translateX(-1.375rem)}to{transform:translateX(3.75rem)}}",
      },
      durations: {
        "signal-in": "400ms",
        "draw-thread": "1.5s",
        "agent-in": "550ms",
        "core-in": "700ms",
        "scroll-line": "2s",
      },
      timingFns: {
        "signal-in": "ease",
        "draw-thread": "cubic-bezier(.65,0,.35,1)",
        "agent-in": "cubic-bezier(.22,1,.36,1)",
        "core-in": "cubic-bezier(.34,1.56,.64,1)",
        "scroll-line": "ease-in-out",
      },
      counts: {
        "scroll-line": "infinite",
      },
    },
  },
  shortcuts: {
    brand: "inline-flex w-max items-center gap-2.5 text-[1.125rem] font-600 tracking-normal",
    "brand-mark": "h-[1.8125rem] w-[1.8125rem] fill-none stroke-current stroke-[1.9] [stroke-linecap:round] [stroke-linejoin:round] [&_.brand-signal]:fill-seal [&_.brand-signal]:stroke-none",
    "mono-label": "font-mono text-[0.6875rem] font-500 tracking-[.17em] text-jade-dark",
    "section-label": "mono-label m-0 flex items-center gap-3",
    "display-heading": "m-0 font-400 tracking-[-.015em] leading-[.98] text-[clamp(3.25rem,6vw,5.75rem)]",
    "script-accent": "font-script text-seal italic font-500 tracking-normal leading-[1]",
    reveal:
      "opacity-100 translate-y-0 transition-[opacity,transform] duration-800 ease-[cubic-bezier(.22,1,.36,1)] motion-reduce:transition-none",
    "agent-card": "absolute flex min-w-[9.875rem] items-center gap-3 border border-ink/20 bg-paper/80 p-2.5 backdrop-blur-[0.5rem] animate-agent-in animate-delay-900 [animation-fill-mode:both] lt-mobile:min-w-0 lt-mobile:p-[0.4375rem]",
    "agent-mark": "grid h-10 w-10 shrink-0 place-items-center bg-ink text-paper lt-mobile:h-8 lt-mobile:w-8",
    "flow-card": "relative min-h-[24.375rem] border-r border-ink/18 pt-[1.9375rem] pr-[4.5vw] lt-mobile:min-h-0 lt-mobile:border-r-0 lt-mobile:border-b lt-mobile:border-ink/18 lt-mobile:px-0 lt-mobile:pt-7 lt-mobile:pb-[3.4375rem]",
  },
  rules: [
    [
      "grain-texture",
      {
        "background-image":
          "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.16'/%3E%3C/svg%3E\")",
      },
    ],
    [
      "hero-backdrop",
      {
        background: "radial-gradient(circle at 79% 38%,rgba(185,214,193,.45),transparent 29%)",
      },
    ],
    [
      "hero-backdrop-stacked",
      {
        background:
          "radial-gradient(circle at 50% 69%,rgba(185,214,193,.45),transparent 28%)",
      },
    ],
  ],
});
