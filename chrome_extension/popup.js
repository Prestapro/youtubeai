const FLASK_PORT = 5001;
document.getElementById('run').addEventListener('click', () => {
  chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
    const videoUrl = tabs[0]?.url || '';
    const data = {
      video: videoUrl,
      keywords: document.getElementById('keywords').value,
      partial: document.getElementById('partial').checked,
      padding_before: parseFloat(document.getElementById('padding_before').value),
      padding_after: parseFloat(document.getElementById('padding_after').value)
    };
    document.getElementById('status').textContent = 'Выполняется…';
    fetch(`http://localhost:${FLASK_PORT || 5001}/api/extract`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    })
    .then(response => {
      if (!response.ok) {
        // If not OK, return response text and throw error
        return response.text().then(text => {
          throw new Error(`Server error: ${text}`);
        });
      }
      // Otherwise parse JSON
      return response.json();
    })
    .then(res => {
      document.getElementById('status').textContent = `Готово, ${res.clips.length} клипов`;
      const clipsDiv = document.getElementById('clips');
      clipsDiv.innerHTML = '';
      res.clips.forEach(fn => {
        const link = document.createElement('a');
        link.href = `http://localhost:${FLASK_PORT || 5001}/assets/${fn}`;
        link.textContent = fn;
        link.target = '_blank';
        clipsDiv.appendChild(link);
        clipsDiv.appendChild(document.createElement('br'));
      });
    })
    .catch(e => {
      document.getElementById('status').textContent = 'Ошибка: ' + e.message;
    });
  });
});