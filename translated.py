import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';

void main() {
  final int BUFF_SIZE = 65536;
  final int WIDTH = 400;
  final int FRAMES_TO_COUNT = 20;
  final int FPS = 0;
  final String HOST_IP = '192.168.1.102';
  final int PORT = 9999;

  // Create a UDP socket
  final socket = RawDatagramSocket.bind(HOST_IP, PORT);
  print('Listening at: $HOST_IP:$PORT');

  // Open video capture device
  final vid = VideoCapture(0);
  var st = 0;
  var cnt = 0;

  while (true) {
    // Receive message from client
    final message = socket.receive();
    final clientAddress = message.address;
    final clientPort = message.port;
    print('GOT connection from $clientAddress:$clientPort');

    while (vid.isOpened()) {
      final frame = vid.read();
      // Resize frame
      final resizedFrame = imutils.resize(frame, width: WIDTH);
      // Encode frame to JPEG with 80% quality
      final encodedFrame = cv2.imencode('.jpg', resizedFrame, [cv2.IMWRITE_JPEG_QUALITY, 80]);
      // Encode frame to base64
      final base64EncodedFrame = base64.encode(encodedFrame);
      // Send frame to client
      socket.send(base64EncodedFrame, clientAddress, clientPort);
      // Draw FPS on frame
      frame = cv2.putText(frame, 'FPS: $FPS', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2);
      // Show frame
      cv2.imshow('TRANSMITTING VIDEO', frame);
      // Wait for 1ms for a key press
      final key = cv2.waitKey(1) & 0xFF;
      if (key == ord('q')) {
        // Close socket if 'q' is pressed
        socket.close();
        break;
      }
      if (cnt == FRAMES_TO_COUNT) {
        try {
          FPS = round(FRAMES_TO_COUNT / (time.time() - st));
          st = time.time();
          cnt = 0;
        } catch (e) {
          // Do nothing
        }
      }
      cnt++;
    }
  }
}
