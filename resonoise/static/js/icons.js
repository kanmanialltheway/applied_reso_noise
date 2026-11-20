// Inline SVG icon set (24x24 grid, stroke based).
const P = {
  upload: '<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>',
  mic: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="2"/>',
  play: '<path d="M8 5.5v13l11-6.5z" fill="currentColor"/>',
  pause: '<rect x="6.5" y="5" width="4" height="14" rx="1" fill="currentColor"/><rect x="13.5" y="5" width="4" height="14" rx="1" fill="currentColor"/>',
  back: '<path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/>',
  fwd: '<path d="M21 12a9 9 0 1 1-3-6.7"/><path d="M21 4v5h-5"/>',
  download: '<path d="M12 4v12M7 11l5 5 5-5"/><path d="M4 18v1a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-1"/>',
  trash: '<path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13M10 11v6M14 11v6"/>',
  wave: '<path d="M3 12h2M7 8v8M11 4v16M15 8v8M19 10v4M21 12h0"/>',
  spectrum: '<path d="M4 20V10M9 20V4M14 20v-8M19 20V7"/>',
  sliders: '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>',
  check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
  alert: '<path d="M12 4l9 16H3z"/><path d="M12 10v4M12 17.5h0"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h0"/>',
  library: '<path d="M5 4v16M10 4v16M15 5l4 .8-3 14.4-4-.8z"/>',
  book: '<path d="M5 4h10a3 3 0 0 1 3 3v13H8a3 3 0 0 1-3-3z"/><path d="M5 17a3 3 0 0 1 3-3h10"/>',
  studio: '<rect x="3" y="4" width="18" height="16" rx="3"/><path d="M7 14l2-4 3 6 2-8 3 6"/>',
  file: '<path d="M6 3h8l4 4v14H6z"/><path d="M14 3v4h4M10 15v-3l4-1v3"/>',
  chevron: '<path d="M9 6l6 6-6 6"/>',
  chevronL: '<path d="M15 6l-6 6 6 6"/>',
  x: '<path d="M6 6l12 12M18 6L6 18"/>',
  volume: '<path d="M4 10v4h4l5 4V6L8 10z"/><path d="M16.5 9a4 4 0 0 1 0 6"/>',
  sparkle: '<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8zM19 16l.8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8z"/>',
  filter: '<path d="M4 5h16l-6 7.5V19l-4 1v-7.5z"/>',
  refresh: '<path d="M20 11a8 8 0 0 0-14-4M4 4v4h4M4 13a8 8 0 0 0 14 4M20 20v-4h-4"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  headphones: '<path d="M4 15v-3a8 8 0 0 1 16 0v3"/><rect x="3" y="14" width="4" height="6" rx="1.5"/><rect x="17" y="14" width="4" height="6" rx="1.5"/>',
  brain: '<path d="M9 4a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 2 5 3 3 0 0 0 6 1V4a3 3 0 0 0-3 0zM15 4a3 3 0 0 1 3 3 3 3 0 0 1 2 5 3 3 0 0 1-2 5 3 3 0 0 1-6 1"/>',
};
export function icon(name, size) {
  const s = size ? ` style="width:${size}px;height:${size}px"` : "";
  return `<svg class="i" viewBox="0 0 24 24" aria-hidden="true"${s}>${P[name] || ""}</svg>`;
}
export const LOGO = `<svg class="brand-mark" viewBox="0 0 32 32" aria-hidden="true">
  <rect x="2" y="13" width="4.4" height="9" rx="2.2" fill="#b7a4cc"/>
  <rect x="8.6" y="7" width="4.4" height="21" rx="2.2" fill="#4b1f52"/>
  <rect x="15.2" y="3" width="4.4" height="26" rx="2.2" fill="#6a3573"/>
  <rect x="21.8" y="10" width="4.4" height="14" rx="2.2" fill="#8ca68c"/>
  <rect x="28" y="14" width="2.6" height="6" rx="1.3" fill="#b7a4cc"/></svg>`;
