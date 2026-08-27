#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>

Arduino_LED_Matrix matrix;
const uint8_t FRAME_SIZE = 104;  // 8 rows x 13 cols

uint8_t alert_frame[FRAME_SIZE] = {
    7,7,0,0,0,0,0,0,0,0,0,7,7,
    0,0,7,7,0,0,0,0,0,7,7,0,0,
    0,0,0,0,7,7,0,7,7,0,0,0,0,
    0,0,0,0,0,7,7,7,0,0,0,0,0,
    0,0,0,0,0,7,7,7,0,0,0,0,0,
    0,0,0,0,7,7,0,7,7,0,0,0,0,
    0,0,7,7,0,0,0,0,0,7,7,0,0,
    7,7,0,0,0,0,0,0,0,0,0,7,7
};
uint8_t clear_frame[FRAME_SIZE] = { 0 };

void set_matrix_alert(int state) {
    if (state == 1) matrix.draw(alert_frame);
    else matrix.draw(clear_frame);
    Serial.print("ALERT state=");
    Serial.println(state);
}

void setup() {
    Serial.begin(115200);
    matrix.begin();
    matrix.setGrayscaleBits(3);
    matrix.clear();
    Bridge.begin();
    Bridge.provide("set_matrix_alert", set_matrix_alert);
}

void loop() {}
