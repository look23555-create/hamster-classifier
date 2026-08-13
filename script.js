// ======================================================
// 🐹 AI 倉鼠辨識 APP
// TensorFlow.js + EfficientNetB0
// ======================================================

// TensorFlow.js 模型位置
const MODEL_URL = "./model/model.json";

// ======================================================
// 8 個分類
// ======================================================

const labels = [
    "三線野生色 (Campbell's Normal)",
    "布丁鼠 (Pudding Hamster)",
    "銀狐倉鼠 (Silver Fox Hamster)",
    "黃金鼠 (Golden Hamster)",
    "蜜袋鼯 (Sugar Glider)",
    "龍貓 (Chinchilla)",
    "花枝鼠 (Fancy Rat)",
    "天竺鼠 (Guinea Pig)"
];

// ======================================================
// 全域變數
// ======================================================

let model = null;


// ======================================================
// 取得 HTML 元件
// ======================================================

const imageUpload = document.getElementById("imageUpload");
const preview = document.getElementById("preview");
const predictButton = document.getElementById("predictButton");
const loading = document.getElementById("loading");
const result = document.getElementById("result");


// ======================================================
// 1️⃣ 載入 AI 模型
// ======================================================

async function loadModel() {

    try {

        loading.textContent = "🤖 AI 模型載入中，請稍候...";

        console.log("正在載入 AI 模型...");

        model = await tf.loadLayersModel(MODEL_URL);

        console.log("=================================");
        console.log("✅ AI 模型載入成功！");
        console.log("=================================");

        console.log("模型輸入：", model.inputs);
        console.log("模型輸出：", model.outputs);

        loading.textContent =
            "✅ AI 模型已準備完成，請上傳照片！";

        // 如果使用者已經選擇圖片
        if (preview.src) {
            predictButton.disabled = false;
        }

    } catch (error) {

        console.error("❌ 模型載入失敗：", error);

        loading.textContent =
            "❌ AI 模型載入失敗，請確認 model 資料夾是否正確。";

    }
}


// ======================================================
// 2️⃣ 使用者選擇圖片
// ======================================================

imageUpload.addEventListener("change", function(event) {

    const file = event.target.files[0];

    if (!file) {
        return;
    }

    // 確認是否為圖片
    if (!file.type.startsWith("image/")) {

        alert("請選擇圖片檔案！");

        return;
    }

    const reader = new FileReader();

    reader.onload = function(e) {

        // 顯示圖片
        preview.src = e.target.result;

        preview.style.display = "block";

        // 清除上一個辨識結果
        result.innerHTML = "";

        loading.textContent =
            "📷 圖片已載入，可以開始辨識！";

        // 模型載入完成才可以辨識
        if (model) {
            predictButton.disabled = false;
        }

    };

    reader.readAsDataURL(file);

});


// ======================================================
// 3️⃣ 開始 AI 辨識
// ======================================================

predictButton.addEventListener("click", async function() {

    // 確認模型
    if (!model) {

        alert("AI 模型尚未載入完成，請稍候。");

        return;
    }

    // 確認圖片
    if (!preview.src) {

        alert("請先選擇一張照片。");

        return;
    }

    // 暫時禁止按鈕，避免重複辨識
    predictButton.disabled = true;

    loading.textContent =
        "🔍 AI 正在分析照片，請稍候...";

    result.innerHTML = "";

    try {

        // ==================================================
        // 建立圖片 Tensor
        // ==================================================

        const prediction = tf.tidy(() => {

            // 從 HTML 圖片建立 Tensor
            let tensor = tf.browser.fromPixels(preview);

            // ==================================================
            // 你的 EfficientNetB0 模型需要：
            // 224 × 224 × 3
            // ==================================================

            tensor = tf.image.resizeBilinear(
                tensor,
                [224, 224]
            );

            // 增加 Batch 維度
            // (224,224,3)
            // ↓
            // (1,224,224,3)

            tensor = tensor.expandDims(0);

            // ==================================================
            // ⚠️ 不要在這裡再做 /255
            //
            // 因為你的模型第一層本身就是：
            // Rescaling
            // ==================================================

            return model.predict(tensor);

        });


        // ==================================================
        // 取得 AI 預測機率
        // ==================================================

        const probabilities =
            await prediction.data();

        // Tensor 不再使用後釋放
        prediction.dispose();


        console.log("AI 預測結果：");

        console.log(probabilities);


        // ==================================================
        // 建立分類結果
        // ==================================================

        const results = labels.map((label, index) => {

            return {

                label: label,

                probability:
                    probabilities[index]

            };

        });


        // ==================================================
        // 由高到低排序
        // ==================================================

        results.sort((a, b) => {

            return b.probability -
                   a.probability;

        });


        console.log("排序後結果：");

        console.log(results);


        // ==================================================
        // 顯示 Top 3
        // ==================================================

        let html = `
            <h2>🎯 AI 辨識結果</h2>
        `;


        results
            .slice(0, 3)
            .forEach((item, index) => {

                const percent =
                    (item.probability * 100)
                    .toFixed(2);


                html += `

                    <div class="result-item">

                        <div class="result-label">

                            ${index + 1}.
                            ${item.label}

                            —
                            ${percent}%

                        </div>


                        <div class="progress">

                            <div
                                class="progress-bar"
                                style="width: ${percent}%">
                            </div>

                        </div>

                    </div>

                `;

            });


        // ==================================================
        // 顯示結果
        // ==================================================

        result.innerHTML = html;

        loading.textContent =
            "✅ 辨識完成！";


    } catch (error) {

        console.error(
            "❌ AI 辨識發生錯誤：",
            error
        );

        result.innerHTML = `
            <p>
                ❌ 辨識失敗，請重新選擇照片。
            </p>
        `;

        loading.textContent =
            "❌ 辨識發生錯誤";

    }


    // 恢復按鈕
    predictButton.disabled = false;

});


// ======================================================
// 4️⃣ 網頁開啟時，自動載入模型
// ======================================================

loadModel();
