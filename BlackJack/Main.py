import cv2
import numpy as np
import mss
from ultralytics import YOLO
from collections import Counter
from sklearn.cluster import DBSCAN
import pyautogui
import time
import os
from collections import deque

# --- Get the absolute path to the directory where the script is located ---
# This makes the script runnable from anywhere and solves file path issues.
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- CONFIGURATION ---
MODEL_PATH = os.path.join(script_dir, '..', 'runs', 'detect', 'best_model', 'weights', 'best.pt')
OK_BUTTON_PATH = os.path.join(script_dir, 'ok_button.png')
CONTINUE_BUTTON_PATH = os.path.join(script_dir, 'continue_button.png')
ALL_BETS_BUTTON_PATH = os.path.join(script_dir, 'all_bets_button.png')
SECOND_BUTTON_PATH = os.path.join(script_dir, 'second_button.png')

# --- SCREEN CAPTURE CONFIGURATION ---
DEALER_CAPTURE_ZONE = {"top": 665, "left": 250, "width": 200, "height": 90}
REGION_1_CAPTURE_ZONE = {"top": 890, "left": 460, "width": 270, "height": 90}
REGION_2_CAPTURE_ZONE = {"top": 890, "left": 1430, "width": 270, "height": 90}

# --- DETECTION & CLUSTERING CONFIGURATION ---
CONFIDENCE_THRESHOLD = 0.8
DBSCAN_EPS = 25
TEXT_AREA_HEIGHT = 200
BUTTON_CHECK_INTERVAL = 1
CONFIRMATION_PIXEL_THRESHOLD = 20

# --- AUTOMATION CONFIGURATION ---
ENABLE_AUTOMATION = True
HIGH_COUNT_THRESHOLD = 5
BET_LOCATION = (1560, 1030)  # (x, y) coordinate for placing a bet

# IMPORTANT: These files must exist in the same folder as this script.
HIT_BUTTON_IMAGE = os.path.join(script_dir, 'hitbutton.png')
STAND_BUTTON_IMAGE = os.path.join(script_dir, 'standbutton.png')
DOUBLE_BUTTON_IMAGE = os.path.join(script_dir, 'doublebutton.png')
SPLIT_BUTTON_IMAGE = os.path.join(script_dir, 'splitbutton.png')

# --- NEW: PRE-FLIGHT FILE CHECK ---
# Verify that all necessary image files exist before starting.
print("[INFO] Checking for required image files...")
required_files = [
    OK_BUTTON_PATH,
    HIT_BUTTON_IMAGE,
    STAND_BUTTON_IMAGE,
    DOUBLE_BUTTON_IMAGE,
    SPLIT_BUTTON_IMAGE,
    CONTINUE_BUTTON_PATH,
]
all_files_found = True
for file_path in required_files:
    if not os.path.exists(file_path):
        print(f"[ERROR] Required file not found: {file_path}")
        all_files_found = False

if not all_files_found:
    print("\n[FATAL] One or more required image files are missing.")
    print(
        "Please ensure all button images are saved in the same directory as the script and the filenames are correct.")
    exit()
print("[INFO] All required files found. Starting automation engine.")
# --- END OF PRE-FLIGHT CHECK ---


# --- BASIC STRATEGY CHARTS ---
# H=Hit, S=Stand, D=Double, P=Split, Ds=Double/Stand
STRATEGY_CHART = {
    'hard': {
        # Player Total: [vs. Dealer 2, 3, 4, 5, 6, 7, 8, 9, 10, A]
        17: ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
        16: ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H'],
        15: ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H'],
        14: ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H'],
        13: ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H'],
        12: ['H', 'H', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H'],
        11: ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D'],
        10: ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'H', 'H'],
        9: ['H', 'D', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],
        # Totals 8 and below are always Hit
    },
    'soft': {
        # Player Total (Ace + Other): [vs. Dealer 2, 3, 4, 5, 6, 7, 8, 9, 10, A]
        9: ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],  # A,9
        8: ['S', 'S', 'S', 'S', 'Ds', 'S', 'S', 'S', 'S', 'S'],  # A,8
        7: ['Ds', 'Ds', 'Ds', 'Ds', 'Ds', 'S', 'S', 'H', 'H', 'H'],  # A,7
        6: ['H', 'D', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],  # A,6
        5: ['H', 'H', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],  # A,5
        4: ['H', 'H', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],  # A,4
        3: ['H', 'H', 'H', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],  # A,3
        2: ['H', 'H', 'H', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],  # A,2
    },
    'pairs': {
        # Player Pair: [vs. Dealer 2, 3, 4, 5, 6, 7, 8, 9, 10, A]
        'A': ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
        '10': ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
        '9': ['P', 'P', 'P', 'P', 'P', 'S', 'P', 'P', 'S', 'S'],
        '8': ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
        '7': ['P', 'P', 'P', 'P', 'P', 'P', 'H', 'H', 'H', 'H'],
        '6': ['P', 'P', 'P', 'P', 'P', 'H', 'H', 'H', 'H', 'H'],
        '5': ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'H', 'H'],  # Played as a hard 10
        '4': ['H', 'H', 'H', 'P', 'P', 'H', 'H', 'H', 'H', 'H'],
        '3': ['P', 'P', 'P', 'P', 'P', 'P', 'H', 'H', 'H', 'H'],
        '2': ['P', 'P', 'P', 'P', 'P', 'P', 'H', 'H', 'H', 'H'],
    }
}

# --- LOAD THE MODEL ---
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def check_for_new_round_buttons(cards):
    """
    Checks for buttons like 'Ok' or 'Continue' to start a new round.
    Also resets the automated play state for the new round.
    """
    global auto_play_in_progress

    buttons_to_check = [OK_BUTTON_PATH, CONTINUE_BUTTON_PATH]

    for button_path in buttons_to_check:
        try:
            button_location = pyautogui.locateCenterOnScreen(button_path, confidence=0.8, grayscale=True)
            if button_location and cards == 0:
                button_name = os.path.basename(button_path)
                print(f"---[ AUTO-PLAY ]--- Found '{button_name}'. Clicking to start new round.")
                auto_play_in_progress = False

                pyautogui.click(button_location)
                time.sleep(1)
                pyautogui.press('f5')
                time.sleep(0.5)
                return  # Exit after finding and clicking one button
        except pyautogui.PyAutoGUIException:
            continue  # This button image wasn't found on screen, try the next one
        except Exception as e:
            print(f"An unexpected error occurred while checking for buttons: {e}")

# --- HELPER FUNCTION for reliable button clicking ---
def find_and_click_button(image_path, action_name, confidence=0.8, retries=5, delay=0.3):
    """
    Tries repeatedly to find and click a button image on the screen.
    """
    for i in range(retries):
        try:
            button_location = pyautogui.locateCenterOnScreen(image_path, confidence=confidence, grayscale=True)
            if button_location:
                pyautogui.click(button_location)
                print(f"---[ AUTO-PLAY ]--- Action '{action_name}' taken. Disengaging for this hand.")
                time.sleep(1)
                return True
        except pyautogui.PyAutoGUIException as image_exception:
            # This exception means the file itself was not found on disk.
            print(f"---[ AUTO-PLAY ]--- PyAutoGUI Error: Cannot find image file at '{image_path}'")
            print(f"---[ AUTO-PLAY ]--- Error details: {image_exception}")
            return False

        time.sleep(delay)

    print(f"---[ AUTO-PLAY ]--- Could not find button for '{action_name}' on screen after {retries} tries.")
    return False

def get_numerical_hand_value(cards):
    """Calculates the final numerical value of a hand, handling Aces."""
    strategy_value_map = {'a': 11, 'k': 10, 'q': 10, 'j': 10, '10': 10, '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}
    total = sum(strategy_value_map.get(c, 0) for c in cards)
    num_aces = cards.count('a')
    while total > 21 and num_aces > 0:
        total -= 10
        num_aces -= 1
    return total

# --- REFACTORED AND IMPROVED BASIC STRATEGY ADVISOR ---
def get_strategy_advice(player_cards, dealer_upcard):
    """
    Determines the optimal playable Blackjack move by first finding the pure
    strategy and then applying game rules (e.g., no doubling on 3+ cards).
    """
    card_values = {'a': 11, 'k': 10, 'q': 10, 'j': 10, '10': 10, '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3,
                   '2': 2}

    if not player_cards or not dealer_upcard:
        return ""

    player_cards = [str(c).lower() for c in player_cards]
    dealer_upcard = str(dealer_upcard).lower()

    try:
        dealer_val = card_values.get(dealer_upcard, 0)
        if not 2 <= dealer_val <= 11: return ""
        dealer_index = (dealer_val - 2) if dealer_val <= 10 else 9
    except (ValueError, TypeError):
        return ""

    # --- Step 1: Determine the PURE strategy action code from the charts ---
    action_code = None

    # Priority 1: Check for a pair to split.
    # If the chart says anything other than 'P', we fall through to treat it as a hard/soft total.
    is_pair = len(player_cards) == 2 and player_cards[0] == player_cards[1]
    if is_pair:
        pair_card = player_cards[0]
        pair_key = 'A' if pair_card == 'a' else ('10' if card_values[pair_card] == 10 else pair_card)
        if pair_key in STRATEGY_CHART['pairs']:
            chart_action = STRATEGY_CHART['pairs'][pair_key][dealer_index]
            # Only a 'P' action is final from the pair chart. Others are treated as totals.
            if chart_action == 'P':
                action_code = 'P'

    # If it's not a splittable pair, evaluate as a soft or hard total.
    if action_code is None:
        num_aces = player_cards.count('a')
        hard_total = sum(card_values.get(c, 0) for c in player_cards)

        # Check for soft totals first
        is_soft = num_aces > 0 and hard_total <= 21
        if is_soft:
            soft_total_key = hard_total - 11 - (num_aces - 1) * 10
            if soft_total_key in STRATEGY_CHART['soft']:
                action_code = STRATEGY_CHART['soft'][soft_total_key][dealer_index]

        # If not a soft hand (or not in the soft chart), treat as hard.
        if action_code is None:
            # De-value aces if necessary for hard total calculation
            aces_to_devalue = num_aces
            while hard_total > 21 and aces_to_devalue > 0:
                hard_total -= 10
                aces_to_devalue -= 1

            if hard_total >= 17:
                action_code = 'S'
            elif hard_total <= 8:
                action_code = 'H'
            elif hard_total in STRATEGY_CHART['hard']:
                action_code = STRATEGY_CHART['hard'][hard_total][dealer_index]

    # --- Step 2: Convert the pure action code to a PLAYABLE action based on game rules ---
    if action_code is None:
        return "Stand"  # Fallback

    can_perform_special_action = len(player_cards) == 2

    if action_code == 'P':
        # The logic above ensures we only get here if the hand is a splittable pair.
        return "Split"
    if action_code == 'D':  # This means Double, otherwise Hit
        return "Double" if can_perform_special_action else "Hit"
    if action_code == 'Ds':  # This means Double, otherwise Stand
        return "DoubleOrStand" if can_perform_special_action else "Stand"
    if action_code == 'S':
        return "Stand"
    if action_code == 'H':
        return "Hit"

    return "Stand"  # Final fallback for safety


def handle_automated_play(player_hands, dealer_upcard, current_running_count):
    """
    Manages automated betting and playing. It will continue to play a hand
    until it stands, busts, doubles, or splits.
    """
    global auto_play_in_progress

    if not ENABLE_AUTOMATION:
        return

    # --- Stage 1: Place a bet if conditions are met ---
    if not auto_play_in_progress and current_running_count >= HIGH_COUNT_THRESHOLD and not player_hands:
        print(f"---[ AUTO-PLAY ]--- High count ({current_running_count}). Placing bet.")
        pyautogui.click(BET_LOCATION)
        auto_play_in_progress = True
        time.sleep(2)
        return

    # --- Stage 2: Play the hand if it's active ---
    if auto_play_in_progress and player_hands:
        # --- FIX: Add a guard clause to wait for the dealer's card ---
        # This prevents the bot from acting before all necessary information is available.
        if not dealer_upcard:
            print("---[ AUTO-PLAY ]--- Player hand detected, waiting for dealer's upcard to be confirmed...")
            time.sleep(0.5)  # Small delay to prevent spamming this message
            return
        # --- END OF FIX ---

        player_hand_cards = player_hands[0]['cards']
        player_value = get_numerical_hand_value(player_hand_cards)

        if player_value > 21:
            print(f"---[ AUTO-PLAY ]--- Player busts with {player_value}. Hand finished.")
            auto_play_in_progress = False
            time.sleep(2)
            return

        advice = get_strategy_advice(player_hand_cards, dealer_upcard)
        print(f"---[ AUTO-PLAY ]--- Current hand: {player_hand_cards} (Value: {player_value}). Advice: {advice}")
        time.sleep(1)

        if advice == "Stand":
            print("---[ AUTO-PLAY ]--- Advice is to Stand. Clicking Stand button.")
            if find_and_click_button(STAND_BUTTON_IMAGE, "Stand"):
                auto_play_in_progress = False
            return

        action_image_path = None
        action_taken = False
        can_perform_special_action = len(player_hand_cards) == 2

        if advice == "Hit":
            action_image_path = HIT_BUTTON_IMAGE
        elif (advice == "Double" or advice == "DoubleOrStand") and can_perform_special_action:
            action_image_path = DOUBLE_BUTTON_IMAGE
        elif advice == "Split" and can_perform_special_action:
            action_image_path = SPLIT_BUTTON_IMAGE

        if action_image_path:
            # This block runs if the primary action (Hit, Double, Split) is possible
            print(f"---[ AUTO-PLAY ]--- Advice is to {advice}. Attempting to click button.")
            # Clean up the advice name for logging if it's "DoubleOrStand"
            log_advice = advice.replace("OrStand", "")
            action_taken = find_and_click_button(action_image_path, log_advice)

            if (advice == "Double" or advice == "DoubleOrStand" or advice == "Split") and action_taken:
                auto_play_in_progress = False  # Hand is over after a Double or Split

            if advice == "Hit" and action_taken:
                time.sleep(1.5)  # Wait for the new card to be dealt
        else:
            # This block handles the FALLBACK actions when a special move isn't possible
            if advice == "Double":
                print(f"---[ AUTO-PLAY ]--- Cannot Double with {len(player_hand_cards)} cards. Defaulting to HIT.")
                if find_and_click_button(HIT_BUTTON_IMAGE, "Hit"):
                    time.sleep(1.5)  # Wait for new card
            elif advice == "DoubleOrStand":
                print(f"---[ AUTO-PLAY ]--- Cannot Double with {len(player_hand_cards)} cards. Defaulting to STAND.")
                if find_and_click_button(STAND_BUTTON_IMAGE, "Stand"):
                    auto_play_in_progress = False  # Hand is over
            else:
                # This case handles if advice is empty, or Split with > 2 cards, etc.
                print(f"---[ AUTO-PLAY ]--- Invalid action '{advice}' for current hand state. Defaulting to STAND for safety.")
                if find_and_click_button(STAND_BUTTON_IMAGE, "Stand"):
                    auto_play_in_progress = False  # Hand is over

# --- CARD ANALYSIS FUNCTION ---
def analyze_hand(results, historical_potentials, is_dealer_hand=False):
    """
    Analyzes model results to find card hands. For the dealer, it requires a card
    to be detected in three consecutive frames. For other hands, it confirms
    cards immediately.

    Args:
        results: The raw output from the YOLO model for the current frame.
        historical_potentials (deque): A deque of length 2 containing card detections
                                       from the previous two frames.
        is_dealer_hand (bool): If True, enables the multi-frame confirmation logic.

    Returns:
        A tuple containing:
        - final_hands (list): A list of dictionaries, each representing a confirmed hand.
        - current_potential_cards (list): A list of all cards detected in the current frame,
                                          to be added to the historical deque for the next iteration.
    """

    def find_and_remove_match(target_card, search_list):
        """Helper to find a similar card, removing it from the list to prevent re-use."""
        for i, card in enumerate(search_list):
            if target_card['name'] == card['name']:
                dist = np.sqrt((target_card['center_x'] - card['center_x']) ** 2 +
                               (target_card['center_y'] - card['center_y']) ** 2)
                if dist < CONFIRMATION_PIXEL_THRESHOLD:
                    # Found a match, remove it from the list and return True
                    search_list.pop(i)
                    return True
        return False

    card_values = {'a': 11, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'j': 10, 'q': 10,
                   'k': 10}
    class_names = results[0].names

    # Step 1: Extract all potential cards from the current frame's results
    current_potential_cards = []
    for obj in results[0].boxes.data:
        confidence = float(obj[4])
        if confidence > CONFIDENCE_THRESHOLD:
            box = obj[0:4]
            center_x = float((box[0] + box[2]) / 2)
            center_y = float((box[1] + box[3]) / 2)
            class_id = int(obj[5])
            class_name = class_names.get(class_id, '')
            current_potential_cards.append({'name': class_name, 'center_x': center_x, 'center_y': center_y})

    # Step 2: Conditionally confirm cards
    if is_dealer_hand:
        confirmed_cards = []
        # MODIFICATION: Check for a length of 1 for 2-frame confirmation
        if len(historical_potentials) == 1:
            # Make a copy so we don't alter the original history deque
            potentials_t1 = list(historical_potentials[-1])  # Frame t-1

            for current_card in current_potential_cards:
                # Check for a match in the most recent frame's history (t-1)
                if find_and_remove_match(current_card, potentials_t1):
                    # Confirmed! The card exists across 2 frames.
                    confirmed_cards.append(current_card)
    else:
        # For player hands, all detected cards are considered "confirmed" immediately.
        confirmed_cards = current_potential_cards

    # Step 3: Cluster and calculate value for confirmed cards only
    if not confirmed_cards:
        return [], current_potential_cards

    card_positions = np.array([[card['center_x'], card['center_y']] for card in confirmed_cards])
    clusters = DBSCAN(eps=DBSCAN_EPS, min_samples=1).fit_predict(card_positions)

    hands_by_cluster = {}
    for i, card in enumerate(confirmed_cards):
        cluster_label = clusters[i]
        if cluster_label != -1:
            hands_by_cluster.setdefault(cluster_label, []).append(card['name'])

    final_hands = []
    for cluster_id in sorted(hands_by_cluster.keys()):
        hand_cards = hands_by_cluster[cluster_id]
        num_aces = hand_cards.count('a')
        base_value = sum(card_values.get(c, 0) for c in hand_cards if c != 'a')
        hand_value_display = ""
        if num_aces > 0:
            hard_value = base_value + num_aces
            soft_value = base_value + 11 + (num_aces - 1)
            if soft_value > 21:
                hand_value_display = str(hard_value)
            else:
                hand_value_display = f"{hard_value}/{soft_value}"
        else:
            hand_value_display = str(base_value)
        final_hands.append({'value': hand_value_display, 'cards': hand_cards})

    # Step 4: Return both the final hands and the list of potential cards for the next frame
    return final_hands, current_potential_cards


def running_count_color(running_count):
    if running_count >= 5:
        return (0, 255, 0)  # Green
    elif running_count < 0:
        return (0, 0, 255)  # Red
    else:
        return (0, 255, 255)  # Yellow


# --- DICTIONARY FOR READABLE CARD NAMES ---
CARD_NAME_MAP = {'a': 'A', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9', '10': '10',
                 'j': 'J', 'q': 'Q', 'k': 'K'}

# --- STATE MANAGEMENT VARIABLES ---
last_dealer_frame = None
last_region_1_frame = None
last_region_2_frame = None

# Use deques to store the history of potential cards for multi-frame confirmation.
# maxlen=2 stores the last two frames (t-1, t-2)
dealer_historical_potentials = deque(maxlen=1)
# Player hands don't use historical confirmation, but we can maintain the structure.
region_1_historical_potentials = deque(maxlen=2)
region_2_historical_potentials = deque(maxlen=2)

running_count = 0
all_last_cards = Counter()
display_texts = []
last_button_check_time = 0
auto_play_in_progress = False

print("Starting live detection dashboard...")
print("Press 'q' in the display window to quit.")

WINDOW_NAME = "Blackjack Analysis Dashboard"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# --- MAIN LOOP FOR LIVE DETECTION ---
with mss.mss() as sct:
    while True:
        current_time = time.time()
        if current_time - last_button_check_time > BUTTON_CHECK_INTERVAL:

            dealer_cards_count = len(dealer_historical_potentials[-1]) if dealer_historical_potentials else 0
            region_1_cards_count = len(region_1_historical_potentials[-1]) if region_1_historical_potentials else 0
            cards_on_screen = dealer_cards_count + region_1_cards_count

            check_for_new_round_buttons(cards_on_screen)
            last_button_check_time = current_time

        dealer_frame = np.array(sct.grab(DEALER_CAPTURE_ZONE))
        region_1_frame = np.array(sct.grab(REGION_1_CAPTURE_ZONE))
        region_2_frame = np.array(sct.grab(REGION_2_CAPTURE_ZONE))

        if not np.array_equal(dealer_frame, last_dealer_frame) or \
                not np.array_equal(region_1_frame, last_region_1_frame) or \
                not np.array_equal(region_2_frame, last_region_2_frame):

            dealer_frame_bgr = cv2.cvtColor(dealer_frame, cv2.COLOR_BGRA2BGR)
            region_1_frame_bgr = cv2.cvtColor(region_1_frame, cv2.COLOR_BGRA2BGR)
            region_2_frame_bgr = cv2.cvtColor(region_2_frame, cv2.COLOR_BGRA2BGR)

            dealer_results = model(dealer_frame_bgr, verbose=False)
            region_1_results = model(region_1_frame_bgr, verbose=False)
            region_2_results = model(region_2_frame_bgr, verbose=False)

            dealer_hands, current_dealer_potentials = analyze_hand(dealer_results, dealer_historical_potentials,
                                                                   is_dealer_hand=True)
            region_1_hands, current_region_1_potentials = analyze_hand(region_1_results, region_1_historical_potentials)
            region_2_hands, current_region_2_potentials = analyze_hand(region_2_results, region_2_historical_potentials)

            all_current_cards_list = [card for hand in dealer_hands for card in hand['cards']] + \
                                     [card for hand in region_1_hands for card in hand['cards']]
            all_current_cards = Counter(all_current_cards_list)
            newly_added_cards = all_current_cards - all_last_cards

            for card, count in newly_added_cards.items():
                if card in ['2', '3', '4', '5', '6']:
                    running_count += count
                elif card in ['10', 'j', 'q', 'k', 'a']:
                    running_count -= count

            display_texts = []
            dealer_upcard = dealer_hands[0]['cards'][0] if dealer_hands and dealer_hands[0]['cards'] else None

            handle_automated_play(region_2_hands, dealer_upcard, running_count)

            for i, hand in enumerate(dealer_hands):
                cards_readable = ', '.join([CARD_NAME_MAP.get(c, '?') for c in hand['cards']])
                display_texts.append(f"Dealer Hand : {cards_readable} (Value: {hand['value']})")

            for i, hand in enumerate(region_2_hands):
                cards_readable = ', '.join([CARD_NAME_MAP.get(c, '?') for c in hand['cards']])
                advice = get_strategy_advice(hand['cards'], dealer_upcard) if dealer_upcard else ""
                advice_text = f" (Advice: {advice})" if advice else ""
                display_texts.append(f"Player Hand {i + 1}: {cards_readable} (Value: {hand['value']}){advice_text}")

            all_last_cards = all_current_cards
            last_dealer_frame = dealer_frame
            last_region_1_frame = region_1_frame
            last_region_2_frame = region_2_frame
            # Update the historical data with the latest frame's detections
            dealer_historical_potentials.append(current_dealer_potentials)
            region_1_historical_potentials.append(current_region_1_potentials)
            region_2_historical_potentials.append(current_region_2_potentials)

            dealer_annotated = dealer_results[0].plot()
            region_1_annotated = region_1_results[0].plot()
            region_2_annotated = region_2_results[0].plot()

            d_h, _, _ = dealer_annotated.shape
            r1_h, _, _ = region_1_annotated.shape
            r2_h, _, _ = region_2_annotated.shape
            video_height = max(d_h, r1_h, r2_h) if max(d_h, r1_h, r2_h) > 0 else 100


            def resize_to_common_height(img, height):
                if img.shape[0] == 0: return np.zeros((height, 1, 3), dtype=np.uint8)
                r = height / img.shape[0]
                dim = (int(img.shape[1] * r), height)
                return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


            dealer_annotated = resize_to_common_height(dealer_annotated, video_height)
            region_1_annotated = resize_to_common_height(region_1_annotated, video_height)
            region_2_annotated = resize_to_common_height(region_2_annotated, video_height)

            top_row_display = np.hstack((dealer_annotated, region_1_annotated, region_2_annotated))

            canvas_width = top_row_display.shape[1]
            canvas_height = video_height + TEXT_AREA_HEIGHT
            dashboard = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
            dashboard[0:video_height, 0:canvas_width] = top_row_display

            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.7
            font_color = (255, 255, 255)
            thickness = 2
            y_pos = video_height + 30
            for text in display_texts:
                cv2.putText(dashboard, text, (10, y_pos), font, font_scale, font_color, thickness)
                y_pos += 30

            running_count_text_color = running_count_color(running_count)
            cv2.putText(dashboard, f"Running Count: {running_count}", (10, y_pos + 15), font, 0.8,
                        running_count_text_color, thickness)

            cv2.imshow(WINDOW_NAME, dashboard)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()