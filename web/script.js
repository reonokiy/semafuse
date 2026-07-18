const revealObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        revealObserver.unobserve(entry.target);
      }
    });
  },
  { threshold: .15 },
);

document.querySelectorAll(".reveal").forEach((element) => revealObserver.observe(element));

const copyButton = document.querySelector(".copy-button");
const commands = `cargo install --path . --locked
semafuse init
semafuse import --agent amp
semafuse import --agent codex
semafuse import --agent claude-code
semafuse import --agent opencode
semafuse search "why did we choose SQLite?"`;

copyButton?.addEventListener("click", async () => {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(commands);
    } else {
      throw new Error("Clipboard API unavailable");
    }
    const label = copyButton.querySelector("span");
    label.textContent = "Copied";
    window.setTimeout(() => {
      label.textContent = "Copy";
    }, 1800);
  } catch {
    const textarea = document.createElement("textarea");
    textarea.value = commands;
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    const copied = document.execCommand("copy");
    textarea.remove();
    const label = copyButton.querySelector("span");
    label.textContent = copied ? "Copied" : "Select text to copy";
    if (copied) {
      window.setTimeout(() => {
        label.textContent = "Copy";
      }, 1800);
    }
  }
});
