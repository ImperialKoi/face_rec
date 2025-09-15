import cv2
import os
import sys
import numpy as np
import time
import textwrap
import math
from pymongo import MongoClient
from dotenv import load_dotenv
from urllib.parse import quote_plus
from deepface import DeepFace
from deepface.modules.verification import find_euclidean_distance
from drag_transform_controller import GestureController

class HolisticUIA:
    def __init__(self):
        print("Initializing Holistic UI Application...")
        self.identified_profile = None
        self.profile_box_state = None
        self.close_button_rect = None
        self.face_present_timer = 0
        self.gesture_controller = GestureController()
        self.setup_window_and_mouse()
        self.load_credentials_and_connect_db()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.prewarm_models()
        self.known_profiles = list(self.profiles_collection.find({"faceDescriptor": {"$exists": True}}))
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def setup_window_and_mouse(self):
        self.WINDOW_NAME = 'Holistic Recognition HUD'
        cv2.namedWindow(self.WINDOW_NAME)
        cv2.setMouseCallback(self.WINDOW_NAME, self.mouse_click_handler)

    def mouse_click_handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.close_button_rect:
            if self.is_point_in_rect((x, y), self.close_button_rect): self.close_profile_box()

    def is_point_in_rect(self, point, rect):
        x, y = point; rx, ry, r_end_x, r_end_y = rect
        return rx <= x <= r_end_x and ry <= y <= r_end_y

    def close_profile_box(self):
        print("[UI] Profile overlay closed.")
        self.identified_profile = None; self.profile_box_state = None; self.close_button_rect = None

    def analyze_face_and_update_state(self, face_roi, face_coords):
        print("\n[Auto-Search] Analyzing new face...")
        try:
            embedding_objs = DeepFace.represent(face_roi, model_name='VGG-Face', enforce_detection=False)
            match = self.find_matching_face(embedding_objs[0]['embedding'], self.known_profiles)
            if match: self.identified_profile = match
            else:
                sim_url = self.simulate_linkedin_reverse_search(); new_profile = self.create_new_profile(sim_url['linkedinUrl'], face_roi)
                if new_profile: self.known_profiles.append(new_profile); self.identified_profile = new_profile
            
            if self.identified_profile:
                print(f"--- PROFILE IDENTIFIED: {self.identified_profile['name']} ---")
                x, y, w, h = face_coords
                initial_pos = [x + w + 200, y + h//2]
                self.profile_box_state = {'pos': initial_pos, 'scale': 1.0, 'rotation': 0, 'is_detached': False, 'state': 'idle', 'controlled_by': None}
        except Exception as e: print(f"Error during face analysis: {e}")

    def run(self):
        print("\n--- Holistic UI Activated ---")
        currently_analyzing = False

        while True:
            ret, frame = self.cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
            
            largest_face_coords = None
            if len(faces) > 0:
                self.face_present_timer = time.time()
                largest_face_coords = max(faces, key=lambda f: f[2] * f[3])
                if not self.identified_profile and not currently_analyzing: currently_analyzing = True
            elif self.profile_box_state and not self.profile_box_state.get('is_detached', False):
                if time.time() - self.face_present_timer > 2: self.close_profile_box()

            hand_data, click_events = self.gesture_controller.process_and_draw_hands(frame)
            if self.profile_box_state:
                self.profile_box_state = self.gesture_controller.handle_interactions(self.profile_box_state, hand_data)
            
            for click in click_events:
                if self.close_button_rect and self.is_point_in_rect(click['pos'], self.close_button_rect):
                    self.close_profile_box()

            if self.identified_profile and self.profile_box_state:
                if not self.profile_box_state['is_detached'] and largest_face_coords is not None:
                    x, y, w, h = largest_face_coords
                    self.profile_box_state['pos'] = [x + w + 200, y + h//2]
                self.draw_profile_overlay(frame, self.identified_profile, self.profile_box_state)
            elif currently_analyzing and largest_face_coords is not None:
                x, y, w, h = largest_face_coords; cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, "ANALYZING...", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            
            cv2.imshow(self.WINDOW_NAME, frame)

            if currently_analyzing:
                x, y, w, h = largest_face_coords
                self.analyze_face_and_update_state(frame[y:y+h, x:x+w], largest_face_coords)
                currently_analyzing = False

            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
        self.cap.release(); cv2.destroyAllWindows(); self.client.close()

    def draw_profile_overlay(self, frame, profile, box_state):
        frame_h, frame_w, _ = frame.shape
        # Create a transparent RGBA layer to draw the UI elements on
        overlay = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)

        center = tuple(map(int, box_state['pos']))
        scale = box_state['scale']
        angle = box_state['rotation']
        
        # Define UI dimensions
        BOX_WIDTH, BOX_HEIGHT, PADDING = int(350 * scale), int(160 * scale), int(15 * scale)
        
        # --- Step 1: Draw all UI elements onto the transparent overlay in their UN-ROTATED positions ---
        # Calculate the top-left corner of the un-rotated box, centered on `center`
        box_tl_x = center[0] - BOX_WIDTH // 2
        box_tl_y = center[1] - BOX_HEIGHT // 2
        
        # Draw the main box background
        cv2.rectangle(overlay, (box_tl_x, box_tl_y), (box_tl_x + BOX_WIDTH, box_tl_y + BOX_HEIGHT), (255, 170, 0, 180), -1)

        # Draw Name (Centered)
        name_font_scale = 0.7 * scale
        (name_w, name_h), _ = cv2.getTextSize(profile['name'], cv2.FONT_HERSHEY_SIMPLEX, name_font_scale, 2)
        name_pos = (center[0] - name_w // 2, box_tl_y + int(35 * scale))
        cv2.putText(overlay, profile['name'], name_pos, cv2.FONT_HERSHEY_SIMPLEX, name_font_scale, (255,255,255,255), 2, cv2.LINE_AA)
        
        # Draw Headline (Left-aligned)
        headline = profile['rawData'].get('headline', '')
        wrapped_lines = textwrap.wrap(headline, width=40)
        headline_font_scale = 0.5 * scale
        y_offset = box_tl_y + int(65 * scale)
        for line in wrapped_lines[:3]:
            line_pos = (box_tl_x + PADDING, y_offset)
            cv2.putText(overlay, line, line_pos, cv2.FONT_HERSHEY_SIMPLEX, headline_font_scale, (255,255,255,255), 1, cv2.LINE_AA)
            y_offset += int(20 * scale)

        # Draw Close Button and calculate its clickable area
        btn_size = int(25 * scale)
        btn_tl_x = box_tl_x + BOX_WIDTH - btn_size - int(10 * scale)
        btn_tl_y = box_tl_y + int(10 * scale)
        
        cv2.rectangle(overlay, (btn_tl_x, btn_tl_y), (btn_tl_x + btn_size, btn_tl_y + btn_size), (0, 0, 180, 255), -1)
        # Add an 'X' to the button for clarity
        x_font_scale = 0.6 * scale
        (x_w, x_h), _ = cv2.getTextSize("X", cv2.FONT_HERSHEY_SIMPLEX, x_font_scale, 2)
        x_pos = (btn_tl_x + (btn_size - x_w) // 2, btn_tl_y + (btn_size + x_h) // 2)
        cv2.putText(overlay, "X", x_pos, cv2.FONT_HERSHEY_SIMPLEX, x_font_scale, (0,0,0,0), 2, cv2.LINE_AA)

        # --- Step 2: Rotate the entire overlay ---
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        rotated_overlay = cv2.warpAffine(overlay, M, (frame_w, frame_h))
        
        # --- Step 3: Blend the rotated overlay with the main frame ---
        alpha_s = rotated_overlay[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(0, 3):
            frame[:, :, c] = (alpha_s * rotated_overlay[:, :, c] + alpha_l * frame[:, :, c])

        # --- Step 4: Calculate the final, rotated bounding box for the close button for clicking ---
        # Define the 4 corners of the button in its original, un-rotated position
        unrotated_btn_pts = np.float32([
            [btn_tl_x, btn_tl_y],
            [btn_tl_x + btn_size, btn_tl_y],
            [btn_tl_x, btn_tl_y + btn_size],
            [btn_tl_x + btn_size, btn_tl_y + btn_size]
        ])
        # Use the same transformation matrix M to find where these corners end up
        rotated_btn_pts = cv2.transform(np.array([unrotated_btn_pts]), M)[0]
        # Get the axis-aligned bounding box of the rotated points
        rx, ry, rw, rh = cv2.boundingRect(rotated_btn_pts)
        self.close_button_rect = (rx, ry, rx + rw, ry + rh)


    # All other utility functions are below
    def load_credentials_and_connect_db(self):
        load_dotenv();MONGO_USER = os.getenv('MONGO_USER');MONGO_PASS = os.getenv('MONGO_PASS');MONGO_CLUSTER = os.getenv('MONGO_CLUSTER')
        if not all([MONGO_USER, MONGO_PASS, MONGO_CLUSTER]): sys.exit("FATAL: MongoDB credentials not found.")
        escaped_user = quote_plus(MONGO_USER);escaped_pass = quote_plus(MONGO_PASS);MONGODB_URI = f"mongodb+srv://{escaped_user}:{escaped_pass}@cluster0.kkdx9xt.mongodb.net/Users?retryWrites=true&w=majority&appName=Cluster0"
        try: self.client = MongoClient(MONGODB_URI);self.client.admin.command('ping');self.db = self.client.get_database();self.profiles_collection = self.db.profiles;print("✅ Successfully connected to MongoDB.")
        except Exception as e: sys.exit(f"FATAL: Error connecting to MongoDB: {e}")
    def prewarm_models(self):
        print("✅ Face detection cascade loaded.");
        try: print("Pre-warming face recognition model..."); DeepFace.represent(np.zeros((100, 100, 3)), model_name='VGG-Face', enforce_detection=False); print("✅ Face recognition model is ready.")
        except Exception as e: print(f"Warning: Could not pre-warm model: {e}")
    def find_matching_face(self, descriptor, profiles):
        best_match, min_dist = None, 0.8
        for p in profiles:
            stored = p.get('faceDescriptor')
            if not stored: continue
            dist = find_euclidean_distance(np.array(descriptor), np.array(stored))
            if dist < min_dist: min_dist, best_match = dist, p.copy(); best_match['confidence'] = round(max(0, (1 - dist / 1.2) * 100), 2)
        return best_match
    def simulate_linkedin_reverse_search(self): return {"linkedinUrl": f"https://www.linkedin.com/in/person-of-interest-{int(time.time() % 10000)}"}
    def scrape_linkedin_profile(self, url: str) -> dict:
        name_part = url.strip('/').split('/in/')[-1]
        full_name = ' '.join([n.capitalize() for n in name_part.split('-') if not n.isdigit()])
        return {"linkedinUrl": url, "summary": "...", "rawData": {"full_name": full_name, "headline": "IncubatorHacks Finance | Full-Stack Developer | Reinforcement Learning", "location": "San Francisco, CA"}}
    def create_new_profile(self, linkedin_url: str, face_image: np.ndarray):
        profile_data = self.scrape_linkedin_profile(linkedin_url)
        embedding_objs = DeepFace.represent(face_image, model_name='VGG-Face', enforce_detection=False)
        face_descriptor = embedding_objs[0]['embedding']
        new_profile_doc = {"name": profile_data["rawData"]["full_name"], "linkedinUrl": profile_data["linkedinUrl"], "summary": profile_data["summary"], "rawData": profile_data["rawData"], "faceDescriptor": face_descriptor}
        insert_result = self.profiles_collection.insert_one(new_profile_doc)
        return self.profiles_collection.find_one({'_id': insert_result.inserted_id})

if __name__ == "__main__":
    app = HolisticUIA()
    app.run()