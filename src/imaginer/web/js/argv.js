// Argument editing only: these tokens go directly to the executable, never a shell.
export function parseArgv(text) {
  const args = [];
  let word = '', quote = null, started = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quote) {
      if (ch === quote) quote = null;
      else if (ch === '\\' && quote === '"') {
        // Windows paths keep their backslashes, including UNC prefixes. Only a
        // run immediately before a double quote escapes that quote/delimiter.
        let end = i;
        while (text[end] === '\\') end++;
        const count = end - i;
        if (text[end] === '"') {
          word += '\\'.repeat(Math.floor(count / 2));
          if (count % 2) word += '"';
          else quote = null;
          i = end;
        } else {
          word += '\\'.repeat(count);
          i = end - 1;
        }
      }
      else word += ch;
    } else if (ch === '"' || ch === "'") {
      quote = ch;
      started = true;
    } else if (/\s/.test(ch)) {
      if (started) args.push(word);
      word = '';
      started = false;
    } else {
      word += ch;
      started = true;
    }
  }
  if (quote) throw new Error('Close the quoted argument before saving.');
  if (started) args.push(word);
  return args;
}

export function formatArgv(args) {
  return args.map((arg) => arg && !/[\s'"\\]/.test(arg) ? arg
    : `"${arg.replace(/(\\*)"/g, (_, slashes) => `${slashes}${slashes}\\"`)
      .replace(/(\\+)$/, '$1$1')}"`).join(' ');
}
