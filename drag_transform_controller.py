import cv2
import mediapipe as mp
import math
import time

class GestureController:
    """Manages hand tracking and gesture interactions using your original, robust state machine,
       enhanced with time-based click confirmation for reliability."""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.lm = self.mp_hands.HandLandmark
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.7, min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.grab_buffer = 40
        self.base_pinch_threshold_2d = 35
        self.depth_compensation_factor = 250
        self.GRACE_PERIOD = 0.15
        self.CLICK_DISTANCE_THRESHOLD = 0.06
        self.CLICK_CONFIRM_DURATION = 0.2  # Require holding click for 200ms
        self.transform_data = {}
        self.hand_states = {
            'left': {'was_clicking': False, 'click_start_time': 0, 'click_armed': False},
            'right': {'was_clicking': False, 'click_start_time': 0, 'click_armed': False}
        }

    def distance_2d(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def distance_3d(self, p1_lm, p2_lm):
        return math.sqrt((p1_lm.x - p2_lm.x)**2 + (p1_lm.y - p2_lm.y)**2 + (p1_lm.z - p2_lm.z)**2)

    def get_hand_data(self, hand_landmarks, frame_shape):
        h, w, _ = frame_shape
        thumb_tip_lm = hand_landmarks[self.lm.THUMB_TIP]
        index_tip_lm = hand_landmarks[self.lm.INDEX_FINGER_TIP]
        
        thumb_pos_2d = [int(thumb_tip_lm.x * w), int(thumb_tip_lm.y * h)]
        index_pos_2d = [int(index_tip_lm.x * w), int(index_tip_lm.y * h)]
        dist_2d = self.distance_2d(thumb_pos_2d, index_pos_2d)
        depth_difference = abs(thumb_tip_lm.z - index_tip_lm.z)
        dynamic_threshold = self.base_pinch_threshold_2d + (depth_difference * self.depth_compensation_factor)
        is_pinching = dist_2d < dynamic_threshold
        pinch_center = [(thumb_pos_2d[0] + index_pos_2d[0]) // 2, (thumb_pos_2d[1] + index_pos_2d[1]) // 2]

        thumb_mcp_lm = hand_landmarks[self.lm.THUMB_MCP]
        click_dist_3d = self.distance_3d(thumb_tip_lm, thumb_mcp_lm)
        is_clicking_now = click_dist_3d < self.CLICK_DISTANCE_THRESHOLD
        
        return {
            'is_pinching': is_pinching, 'pinch_pos': pinch_center, 
            'is_clicking_now': is_clicking_now, 'index_tip_pos': index_pos_2d,
            'landmarks': hand_landmarks
        }

    def process_and_draw_hands(self, frame):
        h, w, _ = frame.shape
        hand_data = {}
        click_events = []
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label.lower()
                data = self.get_hand_data(hand_landmarks.landmark, (h, w, 3))
                hand_data[label] = data

                state = self.hand_states[label]
                if data['is_clicking_now'] and not state['was_clicking']:
                    state['click_start_time'] = time.time()
                
                if data['is_clicking_now'] and time.time() - state['click_start_time'] > self.CLICK_CONFIRM_DURATION:
                    if not state['click_armed']:
                        click_events.append({'pos': data['index_tip_pos']})
                        state['click_armed'] = True
                
                if not data['is_clicking_now']:
                    state['click_armed'] = False
                
                state['was_clicking'] = data['is_clicking_now']
                
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                if data['is_pinching']:
                    cv2.circle(frame, tuple(data['pinch_pos']), 15, (0, 255, 0), 3)
                if state['was_clicking']:
                    progress = min(1.0, (time.time() - state['click_start_time']) / self.CLICK_CONFIRM_DURATION)
                    color = (0, 255, 255) if progress < 1.0 else (0, 0, 255)
                    cv2.circle(frame, tuple(data['index_tip_pos']), int(5 + progress * 20), color, -1)
                    
        return hand_data, click_events

    def handle_interactions(self, ui_element, hand_data):
        if not ui_element: return None
        
        current_time = time.time()
        state = ui_element['state']

        def is_pinch_near_box(pinch_pos, box_state):
            box_center = box_state['pos']
            return self.distance_2d(pinch_pos, box_center) < 200 * box_state['scale'] + self.grab_buffer

        controlled_hands = set(ui_element['controlled_by'] if isinstance(ui_element['controlled_by'], list) else [ui_element['controlled_by']]) if ui_element['controlled_by'] else set()
        free_hands = {label: data for label, data in hand_data.items() if label not in controlled_hands}

        if state == 'drag':
            if not hand_data.get(ui_element['controlled_by'], {}).get('is_pinching', False): ui_element.update({'state': 'drag_grace', 'grace_start_time': current_time})
        elif state == 'drag_grace':
            if current_time - ui_element['grace_start_time'] > self.GRACE_PERIOD: ui_element.update({'state': 'idle', 'controlled_by': None})
            elif hand_data.get(ui_element['controlled_by'], {}).get('is_pinching', False): ui_element['state'] = 'drag'
        elif state == 'transform':
            ctrls = ui_element['controlled_by']
            l_pinch = hand_data.get(ctrls[0], {}).get('is_pinching', False); r_pinch = hand_data.get(ctrls[1], {}).get('is_pinching', False)
            if not l_pinch and not r_pinch: ui_element.update({'state': 'idle', 'controlled_by': None})
            elif not l_pinch or not r_pinch: ui_element.update({'state': 'transform_grace', 'grace_start_time': current_time})
        elif state == 'transform_grace':
            ctrls = ui_element['controlled_by']
            if current_time - ui_element['grace_start_time'] > self.GRACE_PERIOD: ui_element.update({'state': 'idle', 'controlled_by': None})
            elif hand_data.get(ctrls[0],{}).get('is_pinching') and hand_data.get(ctrls[1],{}).get('is_pinching'): ui_element['state'] = 'transform'

        if ui_element['state'] == 'drag':
            dragging_hand = ui_element['controlled_by']
            other_hand = 'left' if dragging_hand == 'right' else 'right'
            if other_hand in free_hands and free_hands[other_hand].get('is_pinching') and is_pinch_near_box(free_hands[other_hand]['pinch_pos'], ui_element):
                ui_element.update({'state': 'transform', 'controlled_by': ['left', 'right']}); self.transform_data.pop(id(ui_element), None)
        
        if ui_element['state'] == 'idle':
            pinching_hands = {l: d for l, d in hand_data.items() if d['is_pinching']}
            if len(pinching_hands) >= 2 and 'left' in pinching_hands and 'right' in pinching_hands and \
               is_pinch_near_box(pinching_hands['left']['pinch_pos'], ui_element) and is_pinch_near_box(pinching_hands['right']['pinch_pos'], ui_element):
                ui_element.update({'state': 'transform', 'controlled_by': ['left', 'right']})
            elif len(pinching_hands) == 1:
                label, data = list(pinching_hands.items())[0]
                if is_pinch_near_box(data['pinch_pos'], ui_element): ui_element.update({'state': 'drag', 'controlled_by': label})

        if ui_element['state'] == 'drag':
            pinch_pos = hand_data[ui_element['controlled_by']]['pinch_pos']
            if 'drag_offset' not in ui_element: ui_element['drag_offset'] = [ui_element['pos'][0] - pinch_pos[0], ui_element['pos'][1] - pinch_pos[1]]
            ui_element['pos'] = [pinch_pos[0] + ui_element['drag_offset'][0], pinch_pos[1] + ui_element['drag_offset'][1]]; ui_element['is_detached'] = True
        elif ui_element['state'] == 'transform':
            left = hand_data['left']['pinch_pos']; right = hand_data['right']['pinch_pos']
            if id(ui_element) not in self.transform_data:
                self.transform_data[id(ui_element)] = {
                    'initial_dist': self.distance_2d(left, right) or 1, 'initial_angle': math.atan2(right[1] - left[1], right[0] - left[0]),
                    'initial_scale': ui_element['scale'], 'initial_rotation': ui_element['rotation'],
                }
            info = self.transform_data[id(ui_element)]
            curr_dist = self.distance_2d(left, right); curr_angle = math.atan2(right[1]-left[1], right[0]-left[0])
            ui_element['scale'] = max(0.5, min(2.0, info['initial_scale'] * (curr_dist / info['initial_dist'])))
            ui_element['rotation'] = info['initial_rotation'] + math.degrees(curr_angle - info['initial_angle'])
            ui_element['pos'] = [(left[0] + right[0]) // 2, (left[1] + right[1]) // 2]; ui_element['is_detached'] = True
        else:
            ui_element.pop('drag_offset', None); self.transform_data.pop(id(ui_element), None)
        return ui_element