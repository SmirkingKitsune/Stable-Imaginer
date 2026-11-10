import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

// Browser ES modules need no package.json or JavaScript build step.
const source = await readFile(new URL('../../src/imaginer/web/js/argv.js', import.meta.url), 'utf8');
const { parseArgv, formatArgv } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

test('quoted paths are single arguments and Windows backslashes survive', () => {
  assert.deepEqual(parseArgv('--model-dir "C:/My Models" --offload-to-cpu'),
    ['--model-dir', 'C:/My Models', '--offload-to-cpu']);
  assert.deepEqual(parseArgv(String.raw`--model-dir "C:\My Models\weights"`),
    ['--model-dir', String.raw`C:\My Models\weights`]);
  assert.deepEqual(parseArgv("--model-dir '/home/me/My Models'"),
    ['--model-dir', '/home/me/My Models']);
  assert.deepEqual(parseArgv(String.raw`--model-dir "\\server\My Models\weights"`),
    ['--model-dir', String.raw`\\server\My Models\weights`]);
});

test('existing argument arrays round trip through the editor without changes', () => {
  const args = ['--model-dir', 'C:\\My Models\\', '\\\\server\\share\\Models', '',
    'a"b', "it's a model", '--key=value with spaces', 'line\nbreak'];
  assert.deepEqual(parseArgv(formatArgv(args)), args);
});

test('empty values, adjacent quotes and whitespace remain valid', () => {
  assert.deepEqual(parseArgv('  --name="my model"\t--empty ""  '),
    ['--name=my model', '--empty', '']);
  assert.deepEqual(parseArgv('   '), []);
});

test('unterminated quotes reject saving instead of returning corrupt argv', () => {
  assert.throws(() => parseArgv('--model-dir "C:/My Models'), /Close the quoted argument/);
  assert.throws(() => parseArgv("--name 'unfinished"), /Close the quoted argument/);
});
