require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { OpenAI } = require('openai');

const app = express();

app.use(cors());
app.use(express.json({ limit: '1mb' }));

const PORT = Number(process.env.PORT) || 3001;
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-4o-mini';
const VALID_TASKS = ['trend', 'ideas', 'reels', 'post', 'weekly'];
const VALID_PROMPT_TOOLS = ['general_image', 'midjourney', 'firefly', 'canva', 'gemini', 'kling'];

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || 'missing',
});

function getMissingApiKeys() {
  const missing = [];
  if (!process.env.OPENAI_API_KEY) missing.push('OPENAI_API_KEY');
  if (!process.env.TAVILY_API_KEY) missing.push('TAVILY_API_KEY');
  return missing;
}

function sendError(res, error, fallbackMessage) {
  if (error.status === 429) {
    return res.status(429).json({
      error: 'Rate limit veya bakiye sorunu var. Biraz bekleyin ya da API bütçenizi kontrol edin.',
    });
  }

  if (error.status === 401) {
    return res.status(401).json({
      error: 'API anahtarı geçersiz veya eksik.',
    });
  }

  if (error.status === 404) {
    return res.status(404).json({
      error: 'Model bulunamadı veya bu modele erişiminiz yok.',
    });
  }

  return res.status(500).json({
    error: error.message || fallbackMessage,
  });
}

async function tavilySearch(query, topic = 'general') {
  let lastError;

  const maxAttempts = 3;

  for (let i = 0; i < maxAttempts; i++) {
    try {
      const response = await fetch('https://api.tavily.com/search', {
        method: 'POST',
        signal: AbortSignal.timeout(30000),
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${process.env.TAVILY_API_KEY}`,
        },
        body: JSON.stringify({
          query,
          topic,
          search_depth: 'advanced',
          include_answer: true,
          include_raw_content: false,
          max_results: 5,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        const error = new Error(`Tavily hatası: ${response.status} - ${shortenText(errorText, 200)}`);
        error.status = response.status;
        throw error;
      }

      return await response.json();
    } catch (error) {
      lastError = error;

      // 4xx hataları (429 hariç) tekrar denemekle düzelmez.
      const isClientError = error.status >= 400 && error.status < 500 && error.status !== 429;
      if (isClientError || i === maxAttempts - 1) break;

      await new Promise((resolve) => setTimeout(resolve, 1500 * (i + 1)));
    }
  }

  throw lastError;
}

async function tavilyMultiSearch(queries, topic = 'general') {
  const allResults = [];
  const allAnswers = [];

  const settled = await Promise.allSettled(queries.map((query) => tavilySearch(query, topic)));
  const failures = settled.filter((item) => item.status === 'rejected');

  if (queries.length && failures.length === queries.length) {
    throw failures[0].reason;
  }

  if (failures.length) {
    console.warn(`Tavily: ${failures.length}/${queries.length} sorgu başarısız oldu.`);
  }

  for (let i = 0; i < queries.length; i++) {
    if (settled[i].status !== 'fulfilled') continue;
    const query = queries[i];
    const data = settled[i].value;

    if (data?.answer) {
      allAnswers.push(`Sorgu: ${query}\nÖzet: ${data.answer}`);
    }

    if (Array.isArray(data?.results)) {
      for (const item of data.results) {
        allResults.push(item);
      }
    }
  }

  const uniqueResults = [];
  const seenUrls = new Set();

  for (const item of allResults) {
    const url = item?.url || '';
    if (!url || seenUrls.has(url)) continue;
    seenUrls.add(url);
    uniqueResults.push(item);
  }

  return {
    answer: allAnswers.join('\n\n'),
    results: uniqueResults.slice(0, 8),
  };
}

function detectResearchTopic(text = '') {
  const lowerText = String(text).toLowerCase();

  const newsKeywords = [
    'gündem',
    'haber',
    'son gelişme',
    'son dakika',
    'trend',
    'güncel',
    'yeni çıkan',
    'lansman',
    'bu hafta',
    'bu ay',
    'recent',
    'latest',
    'news',
    'launch',
    'trending',
    String(new Date().getFullYear()),
  ];

  const isNewsFocused = newsKeywords.some((keyword) => lowerText.includes(keyword));
  return isNewsFocused ? 'news' : 'general';
}

function shortenText(text = '', maxLength = 120) {
  return String(text).replace(/\s+/g, ' ').trim().slice(0, maxLength);
}

function cleanText(value, fallback = '-') {
  if (value === null || value === undefined) return fallback;
  const text = String(value).replace(/\s+/g, ' ').trim();
  return text || fallback;
}

function cleanArray(arr) {
  if (!Array.isArray(arr)) return [];
  return arr.map((item) => cleanText(item, '')).filter(Boolean);
}

function formatResearchResults(results) {
  const resultsText = results
    .map((item, index) => {
      return `${index + 1}. Başlık: ${cleanText(item.title)}
URL: ${cleanText(item.url)}
Özet: ${shortenText(item.content, 220)}`;
    })
    .join('\n\n');

  const sources = results.map((item, index) => ({
    index: index + 1,
    title: cleanText(item.title),
    url: cleanText(item.url),
  }));

  return { resultsText, sources };
}

function safeJsonParse(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function extractJson(text) {
  if (!text) {
    throw new Error('Model boş yanıt döndürdü.');
  }

  if (typeof text === 'object') {
    return text;
  }

  const trimmed = String(text).trim();

  const direct = safeJsonParse(trimmed);
  if (direct && typeof direct === 'object' && !Array.isArray(direct)) return direct;

  const codeBlockMatch =
    trimmed.match(/```json\s*([\s\S]*?)```/i) ||
    trimmed.match(/```\s*([\s\S]*?)```/i);

  if (codeBlockMatch?.[1]) {
    const parsed = safeJsonParse(codeBlockMatch[1].trim());
    if (parsed) return parsed;
  }

  const firstBrace = trimmed.indexOf('{');
  const lastBrace = trimmed.lastIndexOf('}');
  if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
    const possibleJson = trimmed.slice(firstBrace, lastBrace + 1);
    const parsed = safeJsonParse(possibleJson);
    if (parsed) return parsed;
  }

  throw new Error('Model çıktısı JSON olarak ayrıştırılamadı.');
}

function buildResearchQueries({ formData, platform, hedef, ton, task, brandProfile }) {
  const brief = cleanText(formData?.aciklama, '');
  const anaUrunler = cleanText(formData?.anaUrunler, '');
  const isletmeTipi = cleanText(formData?.isletmeTipi, '');
  const hedefKitle = cleanText(formData?.hedefKitle, '');
  const markaAdi = cleanText(formData?.markaAdi || brandProfile?.brandName, '');
  const shortPlatform = shortenText(platform, 30) || 'Instagram';
  const shortHedef = shortenText(hedef, 40);
  const shortTon = shortenText(ton, 30);
  const shortTask = shortenText(task, 20);

  const baseTopic = [markaAdi, isletmeTipi, anaUrunler, brief].filter(Boolean).join(' | ');

  const queries = [
    `${baseTopic} ${hedefKitle} hedef kitle için içerikte öne çıkan gerçek ürün/hizmet unsurları`,
    `${baseTopic} ${shortPlatform} için yüksek etkileşim alan içerik formatları`,
    `${baseTopic} ${shortPlatform} reels post carousel içerik açıları`,
    `${baseTopic} ${shortPlatform} trend içerikler çekim akışı storyboard örnekleri`,
  ];

  if (shortHedef) {
    queries.push(`${baseTopic} ${shortPlatform} ${shortHedef} için içerik yaklaşımı`);
  }

  if (shortTon) {
    queries.push(`${baseTopic} marka tonu ${shortTon} içerik dili`);
  }

  if (shortTask) {
    queries.push(`${baseTopic} ${shortTask} için en uygun içerik kurgusu`);
  }

  return queries.map((q) => q.slice(0, 380)).slice(0, 7);
}

function buildBrandResearchQueries({
  brandName,
  websiteUrl,
  googleMapsUrl,
  instagramUrl,
  tiktokUrl,
}) {
  const queries = [];

  if (brandName) {
    queries.push(`${shortenText(brandName, 80)} marka bilgileri logo renkler konum ürünler`);
    queries.push(`${shortenText(brandName, 80)} website instagram menu products location`);
  }

  if (websiteUrl) {
    queries.push(`${shortenText(websiteUrl, 140)} brand logo colors products about contact`);
  }

  if (googleMapsUrl) {
    queries.push(`${shortenText(googleMapsUrl, 140)} location address place details`);
  }

  if (instagramUrl) {
    queries.push(`${shortenText(instagramUrl, 140)} instagram profile brand bio visual style`);
  }

  if (tiktokUrl) {
    queries.push(`${shortenText(tiktokUrl, 140)} tiktok brand profile content style`);
  }

  return queries.filter(Boolean).slice(0, 6);
}

function buildStructuredPrompt(task) {
  const commonRules = `
Sadece geçerli bir JSON object üret.
JSON dışında hiçbir açıklama yazma.
Bütün metinler Türkçe olsun.
Devrik cümle kurma.
Eksik alan bırakma.
Kullanıcının verdiği konuya sadık kal.
Kullanıcı AI istemediyse AI araçlarına kayma.
Uydurma bilgi verme.
Genel ve her sektöre uyabilecek boş tavsiyeler verme.
Önce işletme / marka / ürün tipini doğru anla.

SENİN EN ÖNEMLİ GÖREVİN:
Kullanıcının verdiği konuya göre sadece içerik üretmek değil, önce şu analizi yapmaktır:
1. İşletme türünü belirle
2. Ana ürün veya hizmeti belirle
3. İnsanların neden ilgileneceğini belirle
4. Bu nişte sosyal medyada neyin çalıştığını çıkar
5. En çok etkileşim alan içerik formatlarını belirle
6. Storyboard / çekim mantığını çıkar

Eğer bu analiz yapılmadan içerik üretirsen, çıktı kalitesiz sayılır.

Özel kurallar:
- İçerik fikri üretmeden önce bu nişte en çok etkileşim alan içerik türlerini belirlemek zorundasın.
- Eğer içerik fikirleri gerçek platform davranışına uygun değilse geçersiz sayılır.
- Her içerik fikri kullanıcıyı kaydetmeye, paylaşmaya veya yorum yapmaya itecek kadar spesifik olmalı.
- “marka hikayesi paylaşın”, “takipçilerinizle bağ kurun” gibi genel öneriler yasak.
- Eğer kullanıcı yiyecek, içecek, kafe, restoran, tatlıcı, fırın, kurabiyeci gibi bir işletmeden bahsediyorsa; ürün yakın planları, sunum, doku, lezzet hissi, hazırlanış süreci, mekan deneyimi, servis anı ve imza ürünler mutlaka düşünülmeli.
- Eğer platform Instagram ise görsel çekicilik, kaydetme potansiyeli, paylaşılabilirlik ve kısa dikkat süresi düşünülmeli.
- Eğer platform LinkedIn ise profesyonel içgörü, uzmanlık, veri ve otorite hissi düşünülmeli.
- Post görevinde yalnızca içerik metni değil, görsel üretim mantığı da ver.
- Görsel prompt; sahneyi, objeleri, kompozisyonu, ışığı, stili ve yazı alanını içermeli.
- Eğer post tasarımı gerekiyorsa, görselin üstünde yazacak kısa metni ayrıca belirt.
- Görsel prompt somut ve üretilebilir olmalı.
`;

  if (task === 'post') {
    return `${commonRules}

JSON yapısı tam olarak şu olsun:
{
  "task": "post",
  "businessInsights": {
    "businessType": "string",
    "mainProductsOrServices": ["string", "string", "string"],
    "targetAudienceProfile": "string",
    "contentPillars": ["string", "string", "string", "string"],
    "instagramWhatWorks": ["string", "string", "string"],
    "storyboardPatterns": ["string", "string", "string"],
    "highEngagementContentTypes": ["string", "string", "string"],
    "viralHooks": ["string", "string", "string"]
  },
  "trendAnalysis": [
    { "title": "string", "score": "string", "whyImportant": "string" },
    { "title": "string", "score": "string", "whyImportant": "string" },
    { "title": "string", "score": "string", "whyImportant": "string" }
  ],
  "selectedDirection": {
    "topic": "string",
    "reason": "string"
  },
  "post": {
    "format": "string",
    "idea": "string",
    "whyItWorks": "string",
    "realDetail": "string",
    "visualSuggestion": "string",
    "headline": "string",
    "caption": "string",
    "cta": "string",
    "hashtags": ["string", "string", "string", "string", "string"],
    "tips": ["string", "string", "string", "string"],
    "imageConcept": "string",
    "textOnImage": "string",
    "designStyle": "string",
    "visualPrompt": "string"
  }
}`;
  }

  if (task === 'reels') {
    return `${commonRules}

JSON yapısı tam olarak şu olsun:
{
  "task": "reels",
  "businessInsights": {
    "businessType": "string",
    "mainProductsOrServices": ["string", "string", "string"],
    "targetAudienceProfile": "string",
    "contentPillars": ["string", "string", "string", "string"],
    "instagramWhatWorks": ["string", "string", "string"],
    "storyboardPatterns": ["string", "string", "string"],
    "highEngagementContentTypes": ["string", "string", "string"],
    "viralHooks": ["string", "string", "string"]
  },
  "trendAnalysis": [
    { "title": "string", "score": "string", "whyImportant": "string" },
    { "title": "string", "score": "string", "whyImportant": "string" },
    { "title": "string", "score": "string", "whyImportant": "string" }
  ],
  "selectedDirection": {
    "topic": "string",
    "reason": "string"
  },
  "reels": {
    "hook": "string",
    "hookReason": "string",
    "videoIdea": "string",
    "videoFormat": "string",
    "duration": "string",
    "flow": ["string", "string", "string", "string"],
    "script": "string",
    "shootingEditing": ["string", "string", "string"],
    "caption": "string",
    "cta": "string",
    "hashtags": ["string", "string", "string", "string", "string"],
    "tips": ["string", "string", "string", "string"]
  }
}`;
  }

  if (task === 'ideas') {
    return `${commonRules}

JSON yapısı tam olarak şu olsun:
{
  "task": "ideas",
  "businessInsights": {
    "businessType": "string",
    "mainProductsOrServices": ["string", "string", "string"],
    "targetAudienceProfile": "string",
    "contentPillars": ["string", "string", "string", "string"],
    "instagramWhatWorks": ["string", "string", "string"],
    "storyboardPatterns": ["string", "string", "string"],
    "highEngagementContentTypes": ["string", "string", "string"],
    "viralHooks": ["string", "string", "string"]
  },
  "trendAnalysis": [
    { "title": "string", "score": "string", "whyImportant": "string" },
    { "title": "string", "score": "string", "whyImportant": "string" },
    { "title": "string", "score": "string", "whyImportant": "string" }
  ],
  "selectedDirection": {
    "topic": "string",
    "reason": "string"
  },
  "ideas": [
    {
      "title": "string",
      "whyItWorks": "string",
      "contentType": "string",
      "reference": "string"
    },
    {
      "title": "string",
      "whyItWorks": "string",
      "contentType": "string",
      "reference": "string"
    },
    {
      "title": "string",
      "whyItWorks": "string",
      "contentType": "string",
      "reference": "string"
    }
  ],
  "bestChoice": {
    "title": "string",
    "reason": "string"
  }
}`;
  }

  if (task === 'weekly') {
    return `${commonRules}

JSON yapısı tam olarak şu olsun:
{
  "task": "weekly",
  "businessInsights": {
    "businessType": "string",
    "mainProductsOrServices": ["string", "string", "string"],
    "targetAudienceProfile": "string",
    "contentPillars": ["string", "string", "string", "string"],
    "instagramWhatWorks": ["string", "string", "string"],
    "storyboardPatterns": ["string", "string", "string"],
    "highEngagementContentTypes": ["string", "string", "string"],
    "viralHooks": ["string", "string", "string"]
  },
  "trendAnalysis": [
    { "title": "string", "score": "string", "whyImportant": "string" },
    { "title": "string", "score": "string", "whyImportant": "string" },
    { "title": "string", "score": "string", "whyImportant": "string" }
  ],
  "selectedDirection": {
    "topic": "string",
    "reason": "string"
  },
  "weeklyPlan": [
    { "day": "Pazartesi", "contentType": "string", "topic": "string", "goal": "string", "hook": "string" },
    { "day": "Salı", "contentType": "string", "topic": "string", "goal": "string", "hook": "string" },
    { "day": "Çarşamba", "contentType": "string", "topic": "string", "goal": "string", "hook": "string" },
    { "day": "Perşembe", "contentType": "string", "topic": "string", "goal": "string", "hook": "string" },
    { "day": "Cuma", "contentType": "string", "topic": "string", "goal": "string", "hook": "string" },
    { "day": "Cumartesi", "contentType": "string", "topic": "string", "goal": "string", "hook": "string" },
    { "day": "Pazar", "contentType": "string", "topic": "string", "goal": "string", "hook": "string" }
  ],
  "weeklyNotes": ["string", "string", "string", "string"]
}`;
  }

  return `${commonRules}

JSON yapısı tam olarak şu olsun:
{
  "task": "trend",
  "businessInsights": {
    "businessType": "string",
    "mainProductsOrServices": ["string", "string", "string"],
    "targetAudienceProfile": "string",
    "contentPillars": ["string", "string", "string", "string"],
    "instagramWhatWorks": ["string", "string", "string"],
    "storyboardPatterns": ["string", "string", "string"],
    "highEngagementContentTypes": ["string", "string", "string"],
    "viralHooks": ["string", "string", "string"]
  },
  "trendAnalysis": [
    { "title": "string", "score": "string", "whyImportant": "string" },
    { "title": "string", "score": "string", "whyImportant": "string" },
    { "title": "string", "score": "string", "whyImportant": "string" }
  ],
  "selectedDirection": {
    "topic": "string",
    "reason": "string"
  },
  "summary": ["string", "string", "string"],
  "contentAngles": ["string", "string", "string"]
}`;
}

function formatTrendSection(data) {
  const trendItems = Array.isArray(data.trendAnalysis) ? data.trendAnalysis : [];

  const trendText = trendItems.length
    ? trendItems
        .map((item, index) => {
          return `${index + 1}. KONU / BAŞLIK
Başlık: ${cleanText(item.title)}
Trend puanı: ${cleanText(item.score)}
Neden önemli: ${cleanText(item.whyImportant)}`;
        })
        .join('\n\n')
    : '1. KONU / BAŞLIK\nBaşlık: -\nTrend puanı: -\nNeden önemli: -';

  return `TREND / İÇGÖRÜ ANALİZİ

${trendText}

SEÇİLEN ANA YÖN
Konu: ${cleanText(data.selectedDirection?.topic)}
Neden: ${cleanText(data.selectedDirection?.reason)}`;
}

function formatTrendOutput(data) {
  const summary = cleanArray(data.summary);
  const angles = cleanArray(data.contentAngles);

  return `${formatTrendSection(data)}

ÖZET
- ${summary[0] || '-'}
- ${summary[1] || '-'}
- ${summary[2] || '-'}

İÇERİKTE KULLANILABİLECEK AÇILAR
1. ${angles[0] || '-'}
2. ${angles[1] || '-'}
3. ${angles[2] || '-'}`;
}

function formatIdeasOutput(data) {
  const ideas = Array.isArray(data.ideas) ? data.ideas : [];
  const engagementTypes = cleanArray(data.businessInsights?.highEngagementContentTypes);
  const hooks = cleanArray(data.businessInsights?.viralHooks);

  const ideasText = ideas.length
    ? ideas
        .map((item, index) => {
          return `FİKİR ${index + 1}
Başlık: ${cleanText(item.title)}
Neden dikkat çeker: ${cleanText(item.whyItWorks)}
İçerik tipi: ${cleanText(item.contentType)}
Kullanılabilecek gerçek örnek / veri / referans: ${cleanText(item.reference)}`;
        })
        .join('\n\n')
    : `FİKİR 1
Başlık: -
Neden dikkat çeker: -
İçerik tipi: -
Kullanılabilecek gerçek örnek / veri / referans: -`;

  return `${formatTrendSection(data)}

BU NİŞTE YÜKSEK ETKİLEŞİM ALAN İÇERİK TÜRLERİ
1. ${engagementTypes[0] || '-'}
2. ${engagementTypes[1] || '-'}
3. ${engagementTypes[2] || '-'}

ETKİLİ HOOK ÖRNEKLERİ
1. ${hooks[0] || '-'}
2. ${hooks[1] || '-'}
3. ${hooks[2] || '-'}

İÇERİK FİKİRLERİ

${ideasText}

EN GÜÇLÜ SEÇİM
Başlık: ${cleanText(data.bestChoice?.title)}
Neden seçildi: ${cleanText(data.bestChoice?.reason)}`;
}

function formatReelsOutput(data) {
  const flow = cleanArray(data.reels?.flow);
  const shooting = cleanArray(data.reels?.shootingEditing);
  const hashtags = cleanArray(data.reels?.hashtags);
  const tips = cleanArray(data.reels?.tips);
  const products = cleanArray(data.businessInsights?.mainProductsOrServices);
  const works = cleanArray(data.businessInsights?.instagramWhatWorks);
  const storyboard = cleanArray(data.businessInsights?.storyboardPatterns);
  const hooks = cleanArray(data.businessInsights?.viralHooks);

  return `${formatTrendSection(data)}

İŞLETME / İÇERİK İÇGÖRÜSÜ
İşletme tipi: ${cleanText(data.businessInsights?.businessType)}
Öne çıkarılacak ana ürünler / hizmetler: ${products.length ? products.join(', ') : '-'}

BU NİŞTE INSTAGRAM'DA NE ÇALIŞIR?
1. ${works[0] || '-'}
2. ${works[1] || '-'}
3. ${works[2] || '-'}

ÖNERİLEN STORYBOARD / AKIŞLAR
1. ${storyboard[0] || '-'}
2. ${storyboard[1] || '-'}
3. ${storyboard[2] || '-'}

VİRAL HOOK YAKLAŞIMLARI
1. ${hooks[0] || '-'}
2. ${hooks[1] || '-'}
3. ${hooks[2] || '-'}

REELS İÇERİĞİ
Hook: ${cleanText(data.reels?.hook)}
Bu hook neden çalışır: ${cleanText(data.reels?.hookReason)}
Video fikri: ${cleanText(data.reels?.videoIdea)}
Video formatı: ${cleanText(data.reels?.videoFormat)}
Önerilen süre: ${cleanText(data.reels?.duration)}

Video akışı:
1. ${flow[0] || '-'}
2. ${flow[1] || '-'}
3. ${flow[2] || '-'}
4. ${flow[3] || '-'}

KONUŞMA METNİ
${cleanText(data.reels?.script)}

ÇEKİM / KURGU ÖNERİSİ
1. ${shooting[0] || '-'}
2. ${shooting[1] || '-'}
3. ${shooting[2] || '-'}

CAPTION
${cleanText(data.reels?.caption)}

CTA: ${cleanText(data.reels?.cta)}
Hashtag: ${hashtags.length ? hashtags.join(' ') : '-'}

İÇERİĞİ GÜÇLENDİRECEK PÜF NOKTALAR
1. ${tips[0] || '-'}
2. ${tips[1] || '-'}
3. ${tips[2] || '-'}
4. ${tips[3] || '-'}`;
}

function formatPostOutput(data) {
  const hashtags = cleanArray(data.post?.hashtags);
  const tips = cleanArray(data.post?.tips);
  const products = cleanArray(data.businessInsights?.mainProductsOrServices);
  const pillars = cleanArray(data.businessInsights?.contentPillars);
  const works = cleanArray(data.businessInsights?.instagramWhatWorks);
  const storyboard = cleanArray(data.businessInsights?.storyboardPatterns);
  const engagementTypes = cleanArray(data.businessInsights?.highEngagementContentTypes);
  const hooks = cleanArray(data.businessInsights?.viralHooks);

  return `${formatTrendSection(data)}

İŞLETME / İÇERİK İÇGÖRÜSÜ
İşletme tipi: ${cleanText(data.businessInsights?.businessType)}
Öne çıkarılacak ana ürünler / hizmetler: ${products.length ? products.join(', ') : '-'}
Hedef kitle profili: ${cleanText(data.businessInsights?.targetAudienceProfile)}

BU NİŞTE INSTAGRAM'DA / PLATFORMDA NE ÇALIŞIR?
1. ${works[0] || '-'}
2. ${works[1] || '-'}
3. ${works[2] || '-'}

YÜKSEK ETKİLEŞİM ALAN İÇERİK TÜRLERİ
1. ${engagementTypes[0] || '-'}
2. ${engagementTypes[1] || '-'}
3. ${engagementTypes[2] || '-'}

ÖNERİLEN İÇERİK SÜTUNLARI
1. ${pillars[0] || '-'}
2. ${pillars[1] || '-'}
3. ${pillars[2] || '-'}
4. ${pillars[3] || '-'}

STORYBOARD / KURGU YAKLAŞIMLARI
1. ${storyboard[0] || '-'}
2. ${storyboard[1] || '-'}
3. ${storyboard[2] || '-'}

ETKİLİ HOOK YAKLAŞIMLARI
1. ${hooks[0] || '-'}
2. ${hooks[1] || '-'}
3. ${hooks[2] || '-'}

POST İÇERİĞİ
Post formatı: ${cleanText(data.post?.format)}
Post fikri: ${cleanText(data.post?.idea)}
Bu fikir neden dikkat çeker: ${cleanText(data.post?.whyItWorks)}
Kullanılacak gerçek veri / örnek / detay: ${cleanText(data.post?.realDetail)}
Görsel / tasarım önerisi: ${cleanText(data.post?.visualSuggestion)}
Post üzerinde yazabilecek başlık: ${cleanText(data.post?.headline)}

GÖRSEL KONSEPTİ
${cleanText(data.post?.imageConcept)}

GÖRSEL ÜZERİNDE YAZACAK METİN
${cleanText(data.post?.textOnImage)}

TASARIM STİLİ
${cleanText(data.post?.designStyle)}

GÖRSEL PROMPT
${cleanText(data.post?.visualPrompt)}

CAPTION
${cleanText(data.post?.caption)}

CTA: ${cleanText(data.post?.cta)}
Hashtag: ${hashtags.length ? hashtags.join(' ') : '-'}

İÇERİĞİ GÜÇLENDİRECEK PÜF NOKTALAR
1. ${tips[0] || '-'}
2. ${tips[1] || '-'}
3. ${tips[2] || '-'}
4. ${tips[3] || '-'}`;
}

function formatWeeklyOutput(data) {
  const weeklyPlan = Array.isArray(data.weeklyPlan) ? data.weeklyPlan : [];
  const weeklyNotes = cleanArray(data.weeklyNotes);

  const fallbackDays = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar'];

  const dayBlocks = fallbackDays
    .map((dayName, index) => {
      const item = weeklyPlan[index] || {};
      return `${dayName.toUpperCase()}
İçerik tipi: ${cleanText(item.contentType)}
Konu: ${cleanText(item.topic)}
Amaç: ${cleanText(item.goal)}
Hook fikri: ${cleanText(item.hook)}`;
    })
    .join('\n\n');

  return `${formatTrendSection(data)}

HAFTALIK İÇERİK PLANI

${dayBlocks}

GENEL STRATEJİ NOTLARI
1. ${weeklyNotes[0] || '-'}
2. ${weeklyNotes[1] || '-'}
3. ${weeklyNotes[2] || '-'}
4. ${weeklyNotes[3] || '-'}`;
}

function formatOutputByTask(task, data) {
  switch (task) {
    case 'trend':
      return formatTrendOutput(data);
    case 'ideas':
      return formatIdeasOutput(data);
    case 'reels':
      return formatReelsOutput(data);
    case 'post':
      return formatPostOutput(data);
    case 'weekly':
      return formatWeeklyOutput(data);
    default:
      return formatTrendOutput(data);
  }
}
function buildPromptToolRules(promptTool, task) {
  const toolMap = {
    general_image: 'Genel görsel üretim aracı',
    midjourney: 'Midjourney',
    firefly: 'Adobe Firefly',
    canva: 'Canva AI',
    gemini: 'Gemini / Imagen',
    kling: 'Kling / video aracı',
  };

  const toolName = toolMap[promptTool] || 'Genel üretim aracı';

  if (promptTool === 'midjourney') {
    return `
Sadece geçerli bir JSON object üret.
JSON dışında hiçbir açıklama yazma.
Bütün metinler Türkçe olsun.
Araç: ${toolName}

TEMEL KURAL:
Midjourney görselin içine güvenilir şekilde okunur metin yazmaz.
Bu yüzden prompt içinde yazıyı görsele gömmeye çalışma.
Metni ayrı alanda ver.
Görsel promptu yazısız kompozisyon mantığıyla kur.
Kompozisyonda metin eklemek için boş alan bırak.

Ek zorunlu kurallar:
- Promptta yalnızca briefte veya brandProfile'da karşılığı olan ürün, mekan, hizmet ve atmosfer öğelerini kullan.
- Ürün/hizmet ile ilgisiz obje ekleme.
- Marka yiyecek/içecek odaklıysa elektronik cihaz, robot, laboratuvar, showroom, fuar sahnesi kurma.
- Görselde yazı efekti, okunur tipografi, tabela metni, poster paragrafı üretmeye çalışma.
- Bunun yerine "clean negative space for headline" gibi boş alan mantığı kur.
- Gerçek logo yoksa başka markaya benzeyen logo uydurma.
- Prompt markanın gerçek bağlamında spesifik olsun.

JSON yapısı tam olarak şu olsun:
{
  "tool": "${promptTool}",
  "task": "${task}",
  "imageConcept": "string",
  "textOnImage": "string",
  "designStyle": "string",
  "prompt": "string",
  "notes": [
    "Midjourney için metni görselin içine yazdırma, sonradan ekle.",
    "Kompozisyonda başlık için temiz boş alan bırak.",
    "Gerçek logo yoksa logo üretme, sadece marka estetiğini yansıt."
  ]
}
`;
  }

  if (promptTool === 'gemini' || promptTool === 'firefly' || promptTool === 'canva' || promptTool === 'general_image') {
    return `
Sadece geçerli bir JSON object üret.
JSON dışında hiçbir açıklama yazma.
Bütün metinler Türkçe olsun.
Araç: ${toolName}

Amaç:
Verilen post içeriğine göre bu araçta kullanılabilecek güçlü bir görsel prompt üret.

Ek zorunlu kurallar:
- Promptta yalnızca briefte veya brandProfile'da karşılığı olan ürün, mekan, hizmet ve atmosfer öğelerini kullan.
- Ürün/hizmet ile ilgisiz obje ekleme.
- Marka yiyecek/içecek odaklıysa elektronik cihaz, robot, laboratuvar, showroom, fuar sahnesi kurma.
- Metin overlay gerekiyorsa çok kısa tut: en fazla 2-4 kelime.
- Metin büyük, okunur ve temiz bir boş alanda yer alsın.
- Uzun slogan, paragraf, küçük yazılar üretme.
- Gerçek logo yoksa başka markaya benzeyen logo üretme.
- Logo varsa birebir aynı olduğunu iddia etme, sadece marka kimliğine yakın görünüm kur.
- Prompt stok görsel mantığında genel değil; markanın gerçek bağlamında spesifik olsun.

JSON yapısı tam olarak şu olsun:
{
  "tool": "${promptTool}",
  "task": "${task}",
  "imageConcept": "string",
  "textOnImage": "string",
  "designStyle": "string",
  "prompt": "string",
  "notes": [
    "Metin kısa tutulmalı.",
    "Yazı için temiz ve okunur boş alan bırakılmalı.",
    "Logo belirsizse başka markaya benzeyen logo üretilmemeli."
  ]
}
`;
  }

  if (promptTool === 'kling') {
    return `
Sadece geçerli bir JSON object üret.
JSON dışında hiçbir açıklama yazma.
Bütün metinler Türkçe olsun.
Araç: ${toolName}

Amaç:
Verilen reels içeriğine göre Kling için video odaklı sahne promptları üret.

Ek zorunlu kurallar:
- Kling için tek poster mantığında görsel prompt değil, hareketli sahne akışı üret.
- Sahne 1, sahne 2, sahne 3 birbirine bağlı olsun.
- Kamera hareketi, ürün odağı, ışık, yakın plan ve atmosfer belirtilsin.
- Metin overlay varsa minimum düzeyde düşün, esas odak sahne olsun.
- Marka yiyecek/içecek odaklıysa ürün hazırlığı, servis, doku, mekan ve müşteri deneyimi etrafında kur.
- Teknoloji markası değilse robotik, laboratuvar, fuar, showroom sahneleri kurma.
- Gerçek logo yoksa logo kullanma.

JSON yapısı tam olarak şu olsun:
{
  "tool": "${promptTool}",
  "task": "reels",
  "thumbnailText": "string",
  "thumbnailPrompt": "string",
  "scenePrompts": [
    "string",
    "string",
    "string"
  ],
  "masterVideoPrompt": "string",
  "notes": [
    "Kling için hareket ve sahne geçişleri net olmalı.",
    "Metin değil sahne akışı öncelikli olmalı.",
    "Logo yoksa uydurulmamalı."
  ]
}
`;
  }

  return `
Sadece geçerli bir JSON object üret.
JSON dışında hiçbir açıklama yazma.
Bütün metinler Türkçe olsun.
Araç: ${toolName}

JSON yapısı tam olarak şu olsun:
{
  "tool": "${promptTool}",
  "task": "${task}",
  "imageConcept": "string",
  "textOnImage": "string",
  "designStyle": "string",
  "prompt": "string",
  "notes": [
    "string",
    "string",
    "string"
  ]
}
`;
}

app.post('/generate', async (req, res) => {
  const { task, platform, hedef, ton, formData, brandProfile } = req.body || {};

  const anaBrief = formData?.aciklama || '';
  if (!String(anaBrief).trim()) {
    return res.status(400).json({
      error: 'Ana konu / brief boş olamaz.',
    });
  }

  if (!VALID_TASKS.includes(task)) {
    return res.status(400).json({
      error: `Geçersiz görev tipi. Geçerli değerler: ${VALID_TASKS.join(', ')}`,
    });
  }

  const missingKeys = getMissingApiKeys();
  if (missingKeys.length) {
    return res.status(500).json({
      error: `Sunucuda eksik API anahtarı: ${missingKeys.join(', ')}. backend/.env dosyasını kontrol edin.`,
    });
  }

  try {
    const safeTask = task;
    const researchTopic = detectResearchTopic(
      `${formData?.aciklama || ''} ${hedef || ''} ${platform || ''} ${ton || ''}`
    );

    const tavilyQueries = buildResearchQueries({
      formData,
      platform,
      hedef,
      ton,
      task: safeTask,
      brandProfile,
    });

    const tavilyData = await tavilyMultiSearch(tavilyQueries, researchTopic);
    const answerText = cleanText(tavilyData.answer, '');
    const results = Array.isArray(tavilyData.results) ? tavilyData.results : [];

    const { resultsText, sources } = formatResearchResults(results);

    const messages = [
      {
        role: 'system',
        content: `
Sen Naiad Creative için çalışan gelişmiş bir içerik stratejisti ve AI destekli content engine'sin.
Uzmanlığın yalnızca yapay zeka araçları değildir.
Kullanıcı hangi konuyu verirse, o konu hakkında araştırma yapıp içerik üretirsin.
${buildStructuredPrompt(safeTask)}
`,
      },
      {
        role: 'user',
        content: `
GÖREV: ${cleanText(task)}
PLATFORM: ${cleanText(platform, 'Belirtilmedi')}
HEDEF: ${cleanText(hedef, 'Belirtilmedi')}
TON: ${cleanText(ton, 'Belirtilmedi')}

FORM VERİSİ
Marka adı: ${cleanText(formData?.markaAdi, 'Belirtilmedi')}
İşletme tipi: ${cleanText(formData?.isletmeTipi, 'Belirtilmedi')}
Hedef kitle: ${cleanText(formData?.hedefKitle, 'Belirtilmedi')}
Araştırma modu: ${cleanText(formData?.arastirmaModu, 'Belirtilmedi')}
Ana ürünler / hizmetler: ${cleanText(formData?.anaUrunler, 'Belirtilmedi')}
Asıl brief: ${cleanText(formData?.aciklama, '')}
Özel not: ${cleanText(formData?.ozelNot, 'Yok')}
Odaklar: ${cleanArray(formData?.odaklar).join(', ') || 'Belirtilmedi'}

MARKA PROFİLİ
${brandProfile ? JSON.stringify(brandProfile, null, 2) : 'Yok'}

ARAŞTIRMA ÖZETİ
${answerText || 'Yok'}

KAYNAK NOTLARI
${resultsText || 'Yok'}

Kurallar:
- İçerik doğrudan markanın gerçek ürün/hizmetine dayanmalı.
- Alakasız obje, sektör veya sahne uydurma.
- Eğer kafe ise teknoloji fuarı dili kullanma.
- Eğer teknoloji markası değilse robot, cihaz showroomu, futuristik gadget sahneleri üretme.
- Briefte geçen ana ürün/hizmet neyse içerik onun etrafında dönsün.
`,
      },
    ];

    const completion = await openai.chat.completions.create({
      model: OPENAI_MODEL,
      messages,
      response_format: { type: 'json_object' },
      temperature: 0.2,
      max_tokens: 2200,
    });

    const finishReason = completion.choices?.[0]?.finish_reason;
    const raw = completion.choices?.[0]?.message?.content || '{}';

    if (finishReason === 'length') {
      return res.status(500).json({
        error: 'Model yanıtı yarıda kesildi. Briefi biraz kısaltıp tekrar deneyin.',
      });
    }

    const structuredData = extractJson(raw);
    const post = formatOutputByTask(safeTask, structuredData);

    return res.json({
      post,
      data: structuredData,
      sources,
    });
  } catch (error) {
    console.error('Hata detayı:', error);
    return sendError(res, error, 'Sunucuda bir hata oluştu.');
  }
});

app.post('/generate-prompt', async (req, res) => {
  const {
    task,
    promptTool,
    originalBrief,
    generatedContent,
    platform,
    hedef,
    ton,
    brandProfile,
    logoUrl,
    logoTextHint,
  } = req.body || {};

  if (!generatedContent || !String(generatedContent).trim()) {
    return res.status(400).json({
      error: 'Prompt üretmek için önce içerik üretimi yapılmış olmalı.',
    });
  }

  if (!process.env.OPENAI_API_KEY) {
    return res.status(500).json({
      error: 'Sunucuda eksik API anahtarı: OPENAI_API_KEY. backend/.env dosyasını kontrol edin.',
    });
  }

  try {
    const safeTask = VALID_TASKS.includes(task) ? task : 'post';
    const safeTool = VALID_PROMPT_TOOLS.includes(promptTool) ? promptTool : 'general_image';

    const brandName = cleanText(brandProfile?.brandName, 'Belirtilmedi');
    const logoHint = cleanText(brandProfile?.logoHint, 'Yok');
    const location = cleanText(brandProfile?.location, 'Yok');
    const website = cleanText(brandProfile?.website, 'Yok');
    const colorPalette = cleanArray(brandProfile?.colorPalette);
    const products = cleanArray(brandProfile?.products);
    const visualStyleNotes = cleanArray(brandProfile?.visualStyleNotes);
    const instagram = cleanText(brandProfile?.socialLinks?.instagram, 'Yok');
    const tiktok = cleanText(brandProfile?.socialLinks?.tiktok, 'Yok');
    const googleMaps = cleanText(brandProfile?.socialLinks?.googleMaps, 'Yok');

    const briefText = cleanText(originalBrief?.aciklama, '');
    const anaUrunler = cleanText(originalBrief?.anaUrunler, '');
    const isletmeTipi = cleanText(originalBrief?.isletmeTipi, '');
    const hedefKitle = cleanText(originalBrief?.hedefKitle, '');
    const ozelNot = cleanText(originalBrief?.ozelNot, '');
    const odaklar = cleanArray(originalBrief?.odaklar).join(', ');
    const markaAdi = cleanText(originalBrief?.markaAdi, '');

    const messages = [
      {
        role: 'system',
        content: `
${buildPromptToolRules(safeTool, safeTask)}

Ek zorunlu kurallar:
- Eğer brandProfile verisi geldiyse promptu buna göre yazmak zorundasın.
- Marka adı, ürünler, renk paleti ve görsel stil notları prompta mutlaka yansısın.
- Logo varsa birebir kopyaladığını iddia etme; "logo stiline yakın" veya "marka kimliğini yansıtan" yaklaşım kullan.
- Renk paleti varsa promptta bu renkleri kullan.
- Ürünler varsa prompt doğrudan o ürünlerin etrafında dönsün.
- Genel ve alakasız objeler uydurma.
- Eğer marka bir kafe / tatlıcı / restoran ise robot, teknoloji fuarı, showroom gibi alakasız sahneler üretme.
- Eğer marka analizi güçlü ise promptun ana kaynağı brandProfile olsun.
`,
      },
      {
        role: 'user',
        content: `
Görev tipi: ${cleanText(safeTask)}
Araç: ${cleanText(safeTool)}
Platform: ${cleanText(platform, 'Belirtilmedi')}
Hedef: ${cleanText(hedef, 'Belirtilmedi')}
Ton: ${cleanText(ton, 'Belirtilmedi')}

ORİJİNAL BRIEF
Marka adı: ${markaAdi || 'Belirtilmedi'}
İşletme tipi: ${isletmeTipi || 'Belirtilmedi'}
Hedef kitle: ${hedefKitle || 'Belirtilmedi'}
Ana ürünler / hizmetler: ${anaUrunler || 'Belirtilmedi'}
Asıl brief: ${briefText || 'Belirtilmedi'}
Özel notlar: ${ozelNot || 'Yok'}
Odaklar: ${odaklar || 'Belirtilmedi'}

ÜRETİLEN İÇERİK
${cleanText(generatedContent, '')}

MARKA PROFİLİ
Marka adı: ${brandName}
Website: ${website}
Lokasyon: ${location}
Logo ipucu: ${logoHint}
Logo URL: ${cleanText(logoUrl, 'Yok')}
Logo metin ipucu: ${cleanText(logoTextHint, 'Yok')}
Renk paleti: ${colorPalette.join(', ') || 'Yok'}
Ürünler: ${products.join(', ') || 'Yok'}
Görsel stil notları: ${visualStyleNotes.join(', ') || 'Yok'}
Instagram: ${instagram}
TikTok: ${tiktok}
Google Maps: ${googleMaps}

Kurallar:
- Prompt, markanın gerçek bağlamından çıkmamalı.
- Eğer yiyecek/içecek işletmesiyse sahne ürün, mekan, servis, sunum, doku ve atmosfer etrafında kurulmalı.
- Eğer teknoloji şirketi değilse teknoloji fuarı, robotik laboratuvar, cihaz showroomu gibi sahneler yasak.
- Eğer post ise tek güçlü ana görsel mantığı kur.
- Eğer reels ise sahneler akış halinde ve birbirine bağlı olsun.
- Eğer kullanıcı gerçek logo URL'si vermediyse kesin logo üretmeye çalışma.
- Eğer logo metin ipucu varsa sadece tipografik yön duygusu olarak kullan.
- Başka bir markaya benzeyen rastgele logo, rozet, amblem, bakery mark işareti uydurma.
`,
      },
    ];

    const completion = await openai.chat.completions.create({
      model: OPENAI_MODEL,
      messages,
      response_format: { type: 'json_object' },
      temperature: 0.2,
      max_tokens: 1600,
    });

    const finishReason = completion.choices?.[0]?.finish_reason;
    const raw = completion.choices?.[0]?.message?.content || '{}';

    if (finishReason === 'length') {
      return res.status(500).json({
        error: 'Prompt yanıtı yarıda kesildi. İçeriği biraz kısaltıp tekrar deneyin.',
      });
    }

    const structuredPrompt = extractJson(raw);

    return res.json({
      promptPack: structuredPrompt,
    });
  } catch (error) {
    console.error('Prompt üretim hatası:', error);
    return sendError(res, error, 'Prompt üretiminde bir hata oluştu.');
  }
});

app.post('/analyze-brand', async (req, res) => {
  const { brandName, websiteUrl, googleMapsUrl, instagramUrl, tiktokUrl } = req.body || {};

  const hasAtLeastOneField = [brandName, websiteUrl, googleMapsUrl, instagramUrl, tiktokUrl].some(
    (item) => String(item || '').trim()
  );

  if (!hasAtLeastOneField) {
    return res.status(400).json({
      error: 'Marka analizi için en az bir alan girilmelidir.',
    });
  }

  const missingKeys = getMissingApiKeys();
  if (missingKeys.length) {
    return res.status(500).json({
      error: `Sunucuda eksik API anahtarı: ${missingKeys.join(', ')}. backend/.env dosyasını kontrol edin.`,
    });
  }

  try {
    const queries = buildBrandResearchQueries({
      brandName,
      websiteUrl,
      googleMapsUrl,
      instagramUrl,
      tiktokUrl,
    });

    const researchTopic = detectResearchTopic(
      `${brandName || ''} ${websiteUrl || ''} ${instagramUrl || ''} ${tiktokUrl || ''}`
    );

    const tavilyData = await tavilyMultiSearch(queries, researchTopic);
    const answerText = cleanText(tavilyData.answer, '');
    const results = Array.isArray(tavilyData.results) ? tavilyData.results : [];

    const { resultsText, sources } = formatResearchResults(results);

    const messages = [
      {
        role: 'system',
        content: `
Sadece geçerli bir JSON object üret.
JSON dışında hiçbir açıklama yazma.
Bütün metinler Türkçe olsun.
Uydurma bilgi verme.
Eğer bir alan kesin değilse bunu notlar kısmında belirsiz diye belirt.

JSON yapısı tam olarak şu olsun:
{
  "brandName": "string",
  "website": "string",
  "location": "string",
  "logoHint": "string",
  "colorPalette": ["string", "string", "string"],
  "products": ["string", "string", "string", "string"],
  "visualStyleNotes": ["string", "string", "string", "string"],
  "targetAudienceHint": "string",
  "brandToneHint": "string",
  "socialLinks": {
    "instagram": "string",
    "tiktok": "string",
    "googleMaps": "string"
  },
  "notes": ["string", "string", "string"]
}
`,
      },
      {
        role: 'user',
        content: `
MARKA ANALİZİ GİRDİLERİ
Marka adı: ${cleanText(brandName, 'Belirtilmedi')}
Website: ${cleanText(websiteUrl, 'Yok')}
Google Maps: ${cleanText(googleMapsUrl, 'Yok')}
Instagram: ${cleanText(instagramUrl, 'Yok')}
TikTok: ${cleanText(tiktokUrl, 'Yok')}

ARAŞTIRMA ÖZETİ
${answerText || 'Yok'}

KAYNAK NOTLARI
${resultsText || 'Yok'}

Kurallar:
- Marka adı mümkünse netleştir.
- Logo stili, renk paleti, ürün grupları ve genel görsel dünya hakkında çıkarım yap.
- Emin olmadığın şeyi kesinmiş gibi yazma.
- Social links alanında kullanıcının verdiği linkleri öncelikle koru.
`,
      },
    ];

    const completion = await openai.chat.completions.create({
      model: OPENAI_MODEL,
      messages,
      response_format: { type: 'json_object' },
      temperature: 0.2,
      max_tokens: 1200,
    });

    const finishReason = completion.choices?.[0]?.finish_reason;
    const raw = completion.choices?.[0]?.message?.content || '{}';

    if (finishReason === 'length') {
      return res.status(500).json({
        error: 'Marka analizi yarıda kesildi. Girdileri biraz azaltıp tekrar deneyin.',
      });
    }

    const brandProfile = extractJson(raw);

    if (instagramUrl) {
      brandProfile.socialLinks = brandProfile.socialLinks || {};
      brandProfile.socialLinks.instagram = instagramUrl;
    }

    if (tiktokUrl) {
      brandProfile.socialLinks = brandProfile.socialLinks || {};
      brandProfile.socialLinks.tiktok = tiktokUrl;
    }

    if (googleMapsUrl) {
      brandProfile.socialLinks = brandProfile.socialLinks || {};
      brandProfile.socialLinks.googleMaps = googleMapsUrl;
    }

    if (websiteUrl) {
      brandProfile.website = websiteUrl;
    }

    if (brandName && (!brandProfile.brandName || brandProfile.brandName === '-')) {
      brandProfile.brandName = brandName;
    }

    return res.json({
      brandProfile,
      sources,
    });
  } catch (error) {
    console.error('Marka analiz hatası:', error);
    return sendError(res, error, 'Marka analizi sırasında hata oluştu.');
  }
});

app.get('/health', (req, res) => {
  res.json({
    ok: true,
    message: 'Server çalışıyor.',
  });
});

app.listen(PORT, () => {
  console.log(`Server ${PORT} portunda çalışıyor.`);

  const missingKeys = getMissingApiKeys();
  if (missingKeys.length) {
    console.warn(`Uyarı: eksik API anahtarı: ${missingKeys.join(', ')}`);
  }
});