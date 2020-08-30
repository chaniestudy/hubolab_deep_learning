import paho.mqtt.client as mqtt
import json
import time

prev_person_loc = []
prev_avg_loc = 0
prev_person_size = []
prev_size = 0
alpha = 0.003
beta = 0.005

def parse_message(item, msg):
    sum = 0
    recvData = str(msg.payload.decode("utf-8"))
    jsonData = json.loads(recvData)  # json 데이터를 dict형으로 파싱

    global prev_person_loc, prev_avg_loc, prev_size

    cent_x = (int(jsonData[item]["x_1"]) + int(jsonData[item]["x_2"])) // 2
    size = (((int(jsonData[item]["x_1"]) - int(jsonData[item]["x_2"]))) * (
                int(jsonData[item]["y_1"]) - int(jsonData[item]["y_2"]))) ** 2

    prev_person_loc.append(cent_x)
    prev_person_size.append(size)

    person_loc = prev_person_loc
    person_size = prev_person_size

    if len(prev_person_loc) >= 10:
        person_loc[9] = cent_x
        del person_loc[0]

    for loc in person_loc:
        sum += loc

    avg_loc = sum // len(person_loc)

    sum = 0

    if len(prev_person_size) >= 10:
        person_size[9] = size
        del person_size[0]

    for size in person_size:
        sum += size

    size = sum // len(person_size)

    if size > (1 + alpha) * prev_size:
        print("person is approaching")
        if avg_loc > (1 + beta) * prev_avg_loc:
            print("and moving right\n")
        elif avg_loc > (1 - alpha) * prev_avg_loc and avg_loc < (1 + alpha) * prev_avg_loc:
            print("right in front of you\n")
        else:
            print("and moving left\n")
    elif size > (1 - beta) * prev_size and avg_loc < (1 + beta) * prev_size:
        if avg_loc > (1 + alpha) * prev_avg_loc:
            print("person is moving right\n")
        elif avg_loc > (1 - alpha) * prev_avg_loc and avg_loc < (1 + alpha) * prev_avg_loc:
            print("person has stopped\n")
        else:
            print("person is moving left\n")
    else:
        print("person is leaving")
        if avg_loc > (1 + alpha) * prev_avg_loc:
            print("and moving right\n")
        elif avg_loc > (1 - alpha) * prev_avg_loc and avg_loc < (1 + alpha) * prev_avg_loc:
            print("right opposite of you\n")
        else:
            print("and moving left\n")

    prev_avg_loc = avg_loc
    prev_person_loc = person_loc
    prev_size = size

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("connected OK")
    else:
        print("Bad connection Returned code=", rc)
    client.subscribe('common', 1)


def on_disconnect(client, userdata, flags, rc=0):
    print(str(rc))


def on_subscribe(client, userdata, mid, granted_qos):
    print("subscribed: " + str(mid) + " " + str(granted_qos))


def on_message(client, userdata, msg):
    parse_message("person", msg)
    # parse_message("bottle", msg)




    # print(prev_person_loc)
    # print("Bottle: " + str(jsonData["bottle"]))

    #
    # client.publish('robot', cmd, 1)
    # client.loop(2)


# 새로운 클라이언트 생성
client = mqtt.Client()
# 콜백 함수 설정 on_connect(브로커에 접속), on_disconnect(브로커에 접속중료), on_subscribe(topic 구독),
# on_message(발행된 메세지가 들어왔을 때)
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_subscribe = on_subscribe
client.on_message = on_message
# address : localhost, port: 1883 에 연결
client.on_connect = on_connect
client.connect('localhost', 1883)
# common topic 으로 메세지 발행
client.loop_forever()