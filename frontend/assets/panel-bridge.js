// Legacy graph/editor/configuration views render inside the shared shell.
// These panels never own conversational work or its connection.
if (window.parent !== window) {
  document.documentElement.classList.add('embedded-panel');
  document.addEventListener('click', event => {
    const link = event.target.closest('a[href]');
    if (!link || event.ctrlKey || event.metaKey || event.shiftKey) return;
    const target = new URL(link.href, location.href);
    if (target.origin === location.origin && !target.pathname.startsWith('/api/')) {
      event.preventDefault();
      window.parent.postMessage({type: 'tamagi:navigate', path: target.pathname + target.hash}, location.origin);
    }
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape') window.parent.postMessage({type: 'tamagi:escape'}, location.origin);
  });
}
