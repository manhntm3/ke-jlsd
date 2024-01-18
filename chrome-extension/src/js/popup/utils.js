import "chrome-extension-async";
import Mercury from "@postlight/mercury-parser";

export async function getCurrentUrl() {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  return tabs[0].url;
}

export async function getKeyword(url, htmlCode) {
  Mercury.parse(url, { html: htmlCode, contentType: "text" }).then((result) => {
    title.textContent = result.title;
    const parsed = result.content;
    const resultFetch = fetch("http://127.0.0.1/wiki/get_wiki_keywords/", {
      method: "POST",
      body: JSON.stringify({ data: parsed }),
    })
      .then((response) => response.json())
      .then(data => {
        // Process the received keywords and scores
        const keywords = data.keywords;
        let listHtml = '<ul class="keyword-list">';
        keywords.forEach(([keyword, score]) => {
          listHtml += `<li class="keyword-item"><strong>${keyword}</strong>: <span>${score.toFixed(2)}</span></li>`;
        });
        listHtml += '</ul>';
        // Update the content of the 'keyword' element
        keyword.innerHTML = listHtml;
      })
      .catch((err) => console.log(err));
    
  });
}

export function getTitle(url) {
  return "";
}