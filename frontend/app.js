// ─── State ───
const state = {
    recentScans: JSON.parse(localStorage.getItem('recentScans') || '[]'),
};

// ─── Helpers ───
function renderStars(rating) {
    const full = Math.floor(rating);
    const half = rating % 1 >= 0.5 ? 1 : 0;
    const empty = 5 - full - half;
    let html = '';
    for (let i = 0; i < full; i++) html += '<span class="star star-full">★</span>';
    if (half) html += '<span class="star star-half">★</span>';
    for (let i = 0; i < empty; i++) html += '<span class="star star-empty">☆</span>';
    return html;
}

// ─── DOM Elements ───
const $ = (sel) => document.querySelector(sel);
const scannerSection = $('#scannerSection');
const loadingSection = $('#loadingSection');
const resultSection = $('#resultSection');
const notFoundSection = $('#notFoundSection');
const recentSection = $('#recentSection');
const fileInput = $('#fileInput');
const cameraBtn = $('#cameraBtn');
const galleryBtn = $('#galleryBtn');
const scanAgainBtn = $('#scanAgainBtn');
const scanAgainBtn2 = $('#scanAgainBtn2');
const dropOverlay = $('#dropOverlay');

// Crop modal elements
const cropModal = $('#cropModal');
const cropImage = $('#cropImage');
const cropConfirm = $('#cropConfirm');
const cropCancel = $('#cropCancel');
let cropper = null;
let pendingFile = null;

// ─── Init ───
renderRecentScans();

// ─── Event Listeners ───
cameraBtn.addEventListener('click', () => {
    fileInput.setAttribute('capture', 'environment');
    fileInput.click();
});

galleryBtn.addEventListener('click', () => {
    fileInput.removeAttribute('capture');
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        showCropEditor(e.target.files[0]);
    }
    fileInput.value = '';
});

scanAgainBtn?.addEventListener('click', resetToScanner);
scanAgainBtn2?.addEventListener('click', resetToScanner);

// ─── Crop Editor ───
function showCropEditor(file) {
    pendingFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        cropImage.src = e.target.result;
        cropModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';

        // Initialize cropper after image loads
        cropImage.onload = () => {
            if (cropper) cropper.destroy();
            cropper = new Cropper(cropImage, {
                aspectRatio: NaN, // Free crop
                viewMode: 1,
                autoCropArea: 0.9,
                responsive: true,
                background: false,
            });
        };
    };
    reader.readAsDataURL(file);
}

function closeCropModal() {
    cropModal.classList.add('hidden');
    document.body.style.overflow = '';
    if (cropper) {
        cropper.destroy();
        cropper = null;
    }
    pendingFile = null;
}

function confirmCrop() {
    if (!cropper) return;

    const canvas = cropper.getCroppedCanvas({
        maxWidth: 1200,
        maxHeight: 1600,
        imageSmoothingEnabled: true,
        imageSmoothingQuality: 'high',
    });

    canvas.toBlob((blob) => {
        closeCropModal();
        const croppedFile = new File([blob], pendingFile?.name || 'photo.jpg', {
            type: 'image/jpeg',
            lastModified: Date.now(),
        });
        scanImage(croppedFile);
    }, 'image/jpeg', 0.92);
}

cropConfirm?.addEventListener('click', confirmCrop);
cropCancel?.addEventListener('click', closeCropModal);

// ─── Photo Offer (Not Found) ───
const sendPhotoYes = $('#sendPhotoYes');
const sendPhotoNo = $('#sendPhotoNo');
const photoUploadArea = $('#photoUploadArea');
const photoUploadInput = $('#photoUploadInput');
const photoUploadBtn = $('#photoUploadBtn');
const photoSuccess = $('#photoSuccess');

sendPhotoYes?.addEventListener('click', () => {
    photoUploadArea?.classList.remove('hidden');
    sendPhotoYes.style.display = 'none';
    sendPhotoNo.style.display = 'none';
    // Hide the offer text
    const photoOfferText = $('#photoOfferText');
    if (photoOfferText) photoOfferText.style.display = 'none';
});

sendPhotoNo?.addEventListener('click', () => {
    const photoOffer = $('#photoOffer');
    photoOffer?.classList.add('hidden');
});

photoUploadBtn?.addEventListener('click', () => {
    photoUploadInput?.click();
});

photoUploadInput?.addEventListener('change', async (e) => {
    if (e.target.files.length === 0) return;
    const file = e.target.files[0];
    try {
        const formData = new FormData();
        formData.append('image', file);
        await fetch('/api/submit-photo', { method: 'POST', body: formData });
    } catch (err) {
        console.error('Photo submit error:', err);
    }
    // Hide upload button
    photoUploadBtn?.classList.add('hidden');
    // Show success message
    photoSuccess?.classList.remove('hidden');
    photoUploadInput.value = '';

    // Swap sad animal → happy animal
    const sadAnimal = $('#sadAnimal');
    const happyAnimal = $('#happyAnimal');
    if (sadAnimal) sadAnimal.classList.add('hidden');
    if (happyAnimal) happyAnimal.classList.remove('hidden');

    // Update titles
    const title = $('#notFoundTitle');
    const subtitle = $('#notFoundSubtitle');
    if (title) title.textContent = 'Отлично! Спасибо за помощь!';
    if (subtitle) subtitle.textContent = 'Мы обязательно добавим это вино в каталог';
});

// ─── Drag & Drop ───
let dragCounter = 0;

document.addEventListener('dragenter', (e) => {
    e.preventDefault();
    dragCounter++;
    dropOverlay.classList.add('active');
});

document.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dragCounter--;
    if (dragCounter <= 0) {
        dragCounter = 0;
        dropOverlay.classList.remove('active');
    }
});

document.addEventListener('dragover', (e) => {
    e.preventDefault();
});

document.addEventListener('drop', (e) => {
    e.preventDefault();
    dragCounter = 0;
    dropOverlay.classList.remove('active');

    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
        showCropEditor(files[0]);
    }
});

// ─── Scan Image ───
async function scanImage(file) {
    showSection('loading');

    const loadingTexts = [
        'Анализируем этикетку...',
        'Ищем в каталоге...',
        'Сравниваем изображения...',
    ];
    let textIdx = 0;
    const subtextEl = $('#loadingSubtext');
    const textInterval = setInterval(() => {
        textIdx = (textIdx + 1) % loadingTexts.length;
        if (subtextEl) subtextEl.textContent = loadingTexts[textIdx];
    }, 800);

    try {
        const formData = new FormData();
        formData.append('image', file);

        const res = await fetch('/api/scan', {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) throw new Error('Scan failed');

        const result = await res.json();
        clearInterval(textInterval);

        if (result.in_catalog && result.wine) {
            showResult(result);
            saveToRecent(result.wine, result.confidence);
        } else {
            showNotFound(result);
        }
    } catch (err) {
        clearInterval(textInterval);
        console.error('Scan error:', err);
        showNotFound({ alternatives: [] });
    }
}

// ─── Show Result ───
function showResult(result) {
    const wine = result.wine;
    const card = $('#resultCard');

    card.innerHTML = `
        <div class="wine-hero">
            <img src="${wine.image_url}" alt="${wine.name}" class="wine-hero-img"
                 onerror="this.style.display='none'">
        </div>
        <div class="wine-info">
            <div class="wine-header">
                <h1 class="wine-name">${wine.name}</h1>
                ${wine.rating ? `<div class="wine-rating"><span class="rating-badge">${wine.rating}</span></div>` : ''}
            </div>
            <p class="wine-producer">${wine.producer}</p>

            <div class="wine-ratings">
                <h3 class="wine-ratings-title">Рейтинг вина</h3>
                <div class="wine-ratings-grid">
                    <div class="rating-block">
                        <span class="rating-block-label">Роскачество</span>
                        <div class="rating-score">
                            <span class="rating-score-value">${wine.quality_rating || 4.1}</span>
                            <span class="rating-score-max">/ 5</span>
                        </div>
                    </div>
                    <div class="rating-block">
                        <span class="rating-block-label">Сомелье</span>
                        <div class="rating-score">
                            <span class="rating-score-value">${wine.sommelier_rating || 87}</span>
                            <span class="rating-score-max">/ 100</span>
                        </div>
                    </div>
                    <div class="rating-block">
                        <span class="rating-block-label">Народный</span>
                        <div class="rating-score">
                            <span class="rating-score-value">${wine.people_rating || 4.2}</span>
                            <span class="rating-score-max">/ 5</span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="wine-meta">
                <div class="meta-item">
                    <span class="meta-label">Регион</span>
                    <span class="meta-value">${wine.region || '—'}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Сорт</span>
                    <span class="meta-value">${wine.grape || '—'}</span>
                </div>
                ${wine.year ? `
                <div class="meta-item">
                    <span class="meta-label">Год</span>
                    <span class="meta-value">${wine.year}</span>
                </div>` : ''}
                <div class="meta-item">
                    <span class="meta-label">Тип</span>
                    <span class="meta-value">${wine.category || '—'}</span>
                </div>
                ${wine.alcohol ? `
                <div class="meta-item">
                    <span class="meta-label">Крепость</span>
                    <span class="meta-value">${wine.alcohol}%</span>
                </div>` : ''}
            </div>

            ${wine.description ? `
            <div class="wine-section">
                <h3>Описание</h3>
                <p>${wine.description}</p>
            </div>` : ''}

            ${wine.food_pairing ? `
            <div class="wine-section">
                <h3>С чем подавать</h3>
                <p>${wine.food_pairing}</p>
            </div>` : ''}
        </div>
    `;

    // Add click to open full page
    card.style.cursor = 'pointer';
    card.onclick = () => {
        window.location.href = `/wine/${wine.slug}`;
    };

    showSection('result');
}

// ─── Show Not Found ───
function showNotFound(result) {
    const altContainer = $('#alternatives');
    altContainer.innerHTML = '';
    const photoOffer = $('#photoOffer');
    const photoUploadArea = $('#photoUploadArea');
    const photoSuccess = $('#photoSuccess');
    const photoOfferText = $('#photoOfferText');
    const photoOfferButtons = $('#photoOfferButtons');

    // Reset photo upload state
    photoOffer?.classList.remove('hidden');
    photoUploadArea?.classList.add('hidden');
    photoSuccess?.classList.add('hidden');

    // Reset offer text and buttons visibility
    if (photoOfferText) photoOfferText.style.display = '';
    if (photoOfferButtons) photoOfferButtons.style.display = '';

    // Reset animal: show sad, hide happy
    const sadAnimal = $('#sadAnimal');
    const happyAnimal = $('#happyAnimal');
    if (sadAnimal) sadAnimal.classList.remove('hidden');
    if (happyAnimal) happyAnimal.classList.add('hidden');

    // Reset titles
    const title = $('#notFoundTitle');
    const subtitle = $('#notFoundSubtitle');
    if (title) title.textContent = 'К сожалению, мы не смогли найти это вино';
    if (subtitle) subtitle.textContent = 'Пока это вино отсутствует в нашем каталоге';

    if (result.alternatives && result.alternatives.length > 0) {
        const altTitle = document.createElement('h3');
        altTitle.textContent = 'Похожие вина:';
        altTitle.style.cssText = 'font-family: "Playfair Display", serif; font-size: 16px; margin-bottom: 12px;';
        altContainer.appendChild(altTitle);

        result.alternatives.slice(0, 3).forEach((alt) => {
            const item = document.createElement('a');
            item.href = `/wine/${alt.slug}`;
            item.className = 'alt-item';
            item.innerHTML = `
                <img src="${alt.image_url}" class="alt-img" alt="" onerror="this.style.display='none'">
                <div class="alt-info">
                    <div class="alt-name">${alt.name}</div>
                    <div class="alt-producer">${alt.producer}</div>
                </div>
            `;
            altContainer.appendChild(item);
        });
    }

    showSection('notFound');
}

// ─── Section Management ───
function showSection(name) {
    scannerSection.classList.toggle('hidden', name !== 'scanner');
    loadingSection.classList.toggle('hidden', name !== 'loading');
    resultSection.classList.toggle('hidden', name !== 'result');
    notFoundSection.classList.toggle('hidden', name !== 'notFound');
}

function resetToScanner() {
    showSection('scanner');
}

// ─── Recent Scans ───
function saveToRecent(wine, confidence) {
    const entry = {
        slug: wine.slug,
        name: wine.name,
        producer: wine.producer,
        image_url: wine.image_url,
        confidence,
        timestamp: Date.now(),
    };

    // Remove duplicate
    state.recentScans = state.recentScans.filter((s) => s.slug !== wine.slug);
    state.recentScans.unshift(entry);
    state.recentScans = state.recentScans.slice(0, 10);

    localStorage.setItem('recentScans', JSON.stringify(state.recentScans));
    renderRecentScans();
}

function renderRecentScans() {
    const list = $('#recentList');
    if (!list) return;

    if (state.recentScans.length === 0) {
        list.innerHTML = '<p class="empty-state">Пока нет сканов</p>';
        return;
    }

    list.innerHTML = state.recentScans
        .map(
            (scan) => `
        <a href="/wine/${scan.slug}" class="recent-item">
            <img src="${scan.image_url}" class="recent-img" alt="" onerror="this.style.display='none'">
            <div class="recent-info">
                <div class="recent-name">${scan.name}</div>
                <div class="recent-meta">${scan.producer}</div>
            </div>
        </a>
    `
        )
        .join('');
}
