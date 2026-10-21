/* Shared by the browser and dependency-free Node tests. */
const StyleLibrary = (() => {
  function matchesStyle(style, query, selectedOnly = false, selected = new Set()) {
    if (selectedOnly && !selected.has(style.id)) return false;
    const text = `${style.label} ${style.prompt || ''}`.toLowerCase();
    return query.toLowerCase().trim().split(/\s+/).every(word => text.includes(word));
  }

  function describeStyle(style) {
    if (!style.prompt) return 'Keep the original musical character';
    const genre = style.prompt.match(/\[Genre:\s*([^\]]+)\]/i);
    return genre ? genre[1] : style.prompt.split('\n').find(line => line.trim()) || '';
  }

  return { matchesStyle, describeStyle };
})();
if (typeof module !== 'undefined') module.exports = StyleLibrary;
