import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# YOLOv8モデルを読み込み
model = YOLO("best.pt")

# Streamlitアプリのセットアップ
st.title("物体検出（YOLOv8）")
st.info("カメラを選択して「Start」ボタンを押すと、物体検出が始まります。")

# webrtc_streamerコンポーネントを使用
# このコンポーネントがカメラ選択のUIを自動的に生成
# video_processor_factoryを使用して、フレームごとに処理を適用
# `webrtc_ctx`オブジェクトを取得
webrtc_ctx = webrtc_streamer(
    key="camera-stream",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
)

# 映像処理のループ
if webrtc_ctx.video_receiver:
    st.subheader("ライブ映像と検出結果")
    frame_placeholder = st.empty()
    results_placeholder = st.empty()
    
    while webrtc_ctx.state.playing:
        try:
            # WebRTCストリームからフレームを1つ取得
            # `av.VideoFrame`オブジェクトとして返される
            frame = webrtc_ctx.video_receiver.recv()

            # OpenCV/Numpyで扱える形式に変換（BGR24）
            frame_np = frame.to_ndarray(format="bgr24")
            
            # YOLOv8による物体検出を実行
            # 結果は`annotated_frame`（描画済み画像）と`labels`（テキスト情報）に分割
            results = model(frame_np, stream=False)
            annotated_frame = results[0].plot()

            # Streamlit上に描画済みフレームと検出結果のテキストを表示
            frame_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
            
            # 検出されたオブジェクトの情報を表示
            detections = results[0].boxes.data.tolist()
            results_text = "検出されたオブジェクト:\n"
            if detections:
                for detection in detections:
                    x1, y1, x2, y2, confidence, class_id = detection
                    results_text += f"- クラス: {model.names[int(class_id)]}, 信頼度: {confidence:.2f}\n"
            else:
                results_text += "オブジェクトは検出されませんでした。"
            results_placeholder.text(results_text)

            # `time.sleep`はアプリをブロックするので、リアルタイム性が損なわれる場合があります
            # 代わりに、Streamlitが自動的に再実行するのを待つ方が良いでしょう
            # ここでは便宜上残しますが、本番環境では削除を検討してください
            time.sleep(0.1)

        except Exception as e:
            st.error(f"フレーム処理中にエラーが発生しました: {e}")
            break

# Upload Image機能
st.markdown("---")
st.subheader("画像アップロード")

uploaded_file = st.file_uploader("画像をアップロードしてください...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    st.image(image, caption="アップロードされた画像", use_container_width=True, channels="BGR")
    
    # YOLOv8による物体検出を実行
    results = model(image, stream=False)
    annotated_frame = results[0].plot()

    st.image(annotated_frame, caption="検出結果", use_container_width=True, channels="BGR")
    
    # 検出結果のテキスト表示
    detections = results[0].boxes.data.tolist()
    results_text = "検出されたオブジェクト:\n"
    if detections:
        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection
            results_text += f"- クラス: {model.names[int(class_id)]}, 信頼度: {confidence:.2f}\n"
    else:
        results_text += "オブジェクトは検出されませんでした。"
    st.text(results_text)