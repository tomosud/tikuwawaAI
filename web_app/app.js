/**
 * チワワ vs チクワ AI判定器
 * ONNX.jsを使用したブラウザ内AI推論
 */

class ChihuahuaChikuwaClassifier {
    constructor() {
        this.model = null;
        this.classNames = ['chihuahua', 'chikuwa'];
        this.classNamesJp = ['チワワ', 'チクワ'];
        this.modelPath = './chihuawa_chikuwa_classifier.onnx';
        this.imageSize = 224;
        this.mean = [0.485, 0.456, 0.406];
        this.std = [0.229, 0.224, 0.225];
        
        this.initializeModel();
    }

    async initializeModel() {
        try {
            console.log('🔄 ONNXモデルを読み込み中...');
            
            // ONNX Runtime Web の設定
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
            
            // モデルセッションを作成
            this.model = await ort.InferenceSession.create(this.modelPath);
            console.log('✅ ONNXモデルの読み込み完了');
            
            // モデル情報をログ出力
            console.log('📋 モデル情報:');
            console.log('- 入力:', this.model.inputNames);
            console.log('- 出力:', this.model.outputNames);
            
        } catch (error) {
            console.error('❌ モデル読み込みエラー:', error);
            this.showError('モデルの読み込みに失敗しました。ONNXファイルが正しく配置されているか確認してください。');
        }
    }

    /**
     * 画像を前処理してテンソルに変換
     */
    async preprocessImage(imageElement) {
        return new Promise((resolve) => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            // キャンバスサイズを設定
            canvas.width = this.imageSize;
            canvas.height = this.imageSize;
            
            // 画像をリサイズして描画
            ctx.drawImage(imageElement, 0, 0, this.imageSize, this.imageSize);
            
            // 画像データを取得
            const imageData = ctx.getImageData(0, 0, this.imageSize, this.imageSize);
            const { data } = imageData;
            
            // RGB値を正規化してテンソル形式に変換
            const tensorData = new Float32Array(3 * this.imageSize * this.imageSize);
            
            for (let i = 0; i < this.imageSize * this.imageSize; i++) {
                const pixelIndex = i * 4; // RGBA
                
                // RGB値を取得して正規化 ([0, 255] -> [0, 1])
                const r = data[pixelIndex] / 255.0;
                const g = data[pixelIndex + 1] / 255.0;
                const b = data[pixelIndex + 2] / 255.0;
                
                // ImageNet標準化を適用
                tensorData[i] = (r - this.mean[0]) / this.std[0]; // R channel
                tensorData[this.imageSize * this.imageSize + i] = (g - this.mean[1]) / this.std[1]; // G channel
                tensorData[2 * this.imageSize * this.imageSize + i] = (b - this.mean[2]) / this.std[2]; // B channel
            }
            
            // テンソル形状: [1, 3, 224, 224]
            const tensor = new ort.Tensor('float32', tensorData, [1, 3, this.imageSize, this.imageSize]);
            resolve(tensor);
        });
    }

    /**
     * ソフトマックス関数
     */
    softmax(logits) {
        const maxLogit = Math.max(...logits);
        const expLogits = logits.map(x => Math.exp(x - maxLogit));
        const sumExp = expLogits.reduce((a, b) => a + b, 0);
        return expLogits.map(x => x / sumExp);
    }

    /**
     * 画像分類を実行
     */
    async predict(imageElement) {
        try {
            if (!this.model) {
                throw new Error('モデルが初期化されていません');
            }

            console.log('🔄 画像前処理中...');
            const inputTensor = await this.preprocessImage(imageElement);
            
            console.log('🔄 AI推論実行中...');
            const feeds = {};
            feeds[this.model.inputNames[0]] = inputTensor;
            
            // 推論を実行
            const results = await this.model.run(feeds);
            const outputTensor = results[this.model.outputNames[0]];
            
            // ロジットからソフトマックスで確率を算出
            const logits = Array.from(outputTensor.data);
            const probabilities = this.softmax(logits);
            
            // 予測結果を取得
            const predictedIndex = probabilities.indexOf(Math.max(...probabilities));
            const confidence = probabilities[predictedIndex];
            
            const result = {
                predictedClass: this.classNames[predictedIndex],
                predictedClassJp: this.classNamesJp[predictedIndex],
                confidence: confidence,
                probabilities: {
                    chihuahua: probabilities[0],
                    chikuwa: probabilities[1]
                }
            };
            
            console.log('✅ 推論完了:', result);
            return result;
            
        } catch (error) {
            console.error('❌ 推論エラー:', error);
            throw error;
        }
    }

    /**
     * エラーメッセージを表示
     */
    showError(message) {
        const resultSection = document.getElementById('resultSection');
        const resultText = document.getElementById('resultText');
        const confidenceText = document.getElementById('confidenceText');
        const detailBox = document.getElementById('detailBox');
        
        resultText.textContent = '❌ エラー';
        confidenceText.textContent = '';
        detailBox.textContent = message;
        
        resultSection.style.display = 'block';
    }
}

// アプリケーションクラス
class App {
    constructor() {
        this.classifier = new ChihuahuaChikuwaClassifier();
        this.setupEventListeners();
    }

    setupEventListeners() {
        const uploadArea = document.getElementById('uploadArea');
        const uploadBtn = document.getElementById('uploadBtn');
        const imageInput = document.getElementById('imageInput');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const resetBtn = document.getElementById('resetBtn');

        // ファイル選択ボタン - アップロードエリアのクリックと重複を避けるため停止伝播
        uploadBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            imageInput.click();
        });

        // アップロードエリアクリック（ボタン以外の領域）
        uploadArea.addEventListener('click', (e) => {
            // ボタンがクリックされた場合は処理しない
            if (e.target !== uploadBtn) {
                imageInput.click();
            }
        });

        // ファイル選択 - 選択後すぐにAI処理を実行
        imageInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleImageUpload(e.target.files[0], true); // 自動実行フラグ
            }
        });

        // ドラッグ&ドロップ - ドロップ後すぐにAI処理を実行
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            if (e.dataTransfer.files.length > 0) {
                this.handleImageUpload(e.dataTransfer.files[0], true); // 自動実行フラグ
            }
        });

        // ペースト機能
        document.addEventListener('paste', (e) => {
            e.preventDefault();
            const items = e.clipboardData.items;
            
            for (let i = 0; i < items.length; i++) {
                const item = items[i];
                if (item.type.startsWith('image/')) {
                    const file = item.getAsFile();
                    if (file) {
                        this.handleImageUpload(file, true); // 自動実行フラグ
                        break;
                    }
                }
            }
        });

        // 判定ボタン
        analyzeBtn.addEventListener('click', () => {
            this.analyzeImage();
        });

        // リセットボタン
        resetBtn.addEventListener('click', () => {
            this.resetApp();
        });
    }

    handleImageUpload(file, autoAnalyze = false) {
        if (!file.type.startsWith('image/')) {
            alert('画像ファイルを選択してください。');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            const previewImage = document.getElementById('previewImage');
            previewImage.src = e.target.result;
            
            // プレビューセクションを表示
            document.getElementById('previewSection').style.display = 'block';
            document.getElementById('resultSection').style.display = 'none';
            
            // 自動実行フラグが設定されている場合、すぐにAI判定を実行
            if (autoAnalyze) {
                // 画像の読み込み完了を待ってから実行
                previewImage.onload = () => {
                    this.analyzeImage();
                };
            }
        };
        reader.readAsDataURL(file);
    }

    async analyzeImage() {
        const previewImage = document.getElementById('previewImage');
        const loadingSection = document.getElementById('loadingSection');
        const resultSection = document.getElementById('resultSection');
        const resultText = document.getElementById('resultText');
        const confidenceText = document.getElementById('confidenceText');
        const detailBox = document.getElementById('detailBox');

        try {
            // ローディング表示
            loadingSection.style.display = 'block';
            resultSection.style.display = 'none';

            // 画像が完全に読み込まれるまで待機
            await new Promise((resolve) => {
                if (previewImage.complete) {
                    resolve();
                } else {
                    previewImage.onload = resolve;
                }
            });

            // AI推論を実行
            const result = await this.classifier.predict(previewImage);

            // 結果を表示
            resultText.textContent = `🎯 ${result.predictedClassJp}`;
            confidenceText.textContent = `信頼度: ${(result.confidence * 100).toFixed(1)}%`;
            
            const detailText = `📊 詳細結果:
• チワワ: ${(result.probabilities.chihuahua * 100).toFixed(1)}%
• チクワ: ${(result.probabilities.chikuwa * 100).toFixed(1)}%

🔍 予測: ${result.predictedClassJp} (${result.predictedClass})
📈 信頼度: ${result.confidence.toFixed(4)} (${(result.confidence * 100).toFixed(2)}%)`;
            
            detailBox.textContent = detailText;

            // ローディングを非表示、結果を表示
            loadingSection.style.display = 'none';
            resultSection.style.display = 'block';

        } catch (error) {
            console.error('分析エラー:', error);
            
            // エラー表示
            resultText.textContent = '❌ 分析失敗';
            confidenceText.textContent = '';
            detailBox.textContent = `エラーが発生しました: ${error.message}`;
            
            loadingSection.style.display = 'none';
            resultSection.style.display = 'block';
        }
    }

    resetApp() {
        // 全セクションをリセット
        document.getElementById('previewSection').style.display = 'none';
        document.getElementById('resultSection').style.display = 'none';
        document.getElementById('loadingSection').style.display = 'none';
        
        // 入力をクリア
        document.getElementById('imageInput').value = '';
        document.getElementById('previewImage').src = '';
    }
}

// アプリケーション初期化
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 チワワ vs チクワ AI判定器を初期化中...');
    new App();
});