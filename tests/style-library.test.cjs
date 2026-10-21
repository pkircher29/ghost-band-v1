const { test } = require('node:test');
const assert = require('node:assert/strict');
const { matchesStyle, describeStyle } = require('../app/static/style-library.js');

test('search matches every word across the name and instrumentation, ignoring case', () => {
  const style = { label: 'Bossa Nova', prompt: '[Genre: Bossa Nova]\nNylon guitar and brushes' };
  assert.equal(matchesStyle(style, '  BOSSA guitar '), true);
  assert.equal(matchesStyle(style, 'bossa trumpet'), false);
  assert.equal(matchesStyle(style, ''), true);
});

test('selected-only combines with search and never changes selections', () => {
  const selected = new Set(['folk']);
  const style = { id: 'folk', label: 'Folk', prompt: 'acoustic guitar' };
  assert.equal(matchesStyle(style, 'guitar', true, selected), true);
  assert.equal(matchesStyle({ ...style, id: 'jazz' }, '', true, selected), false);
  assert.deepEqual([...selected], ['folk']);
});

test('descriptions support the original baseline and custom free-text prompts', () => {
  assert.match(describeStyle({ prompt: '' }), /original/i);
  assert.equal(describeStyle({ prompt: 'Warm piano and brushed drums' }), 'Warm piano and brushed drums');
  assert.equal(describeStyle({ prompt: 'Language: English\n[Genre: Soul / R&B]\n[Tempo: 90 BPM]' }), 'Soul / R&B');
});
