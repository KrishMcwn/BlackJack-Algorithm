import sys
from itertools import combinations, combinations_with_replacement

# --- UNIFIED BASIC STRATEGY CHART ---
# This structure is more robust and easier to verify against a visual chart.
# Actions: H=Hit, S=Stand, D=Double/Hit, Ds=Double/Stand, P=Split
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


# --- Comprehensive Test Runner ---
def run_test_suite(hands_to_test, hand_size, card_values, dealer_card_ranks, action_map):
    """Runs the strategy test for a given set of hands."""
    total_tests = 0
    failed_tests = 0
    failed_cases = []

    for hand in hands_to_test:
        # Skip hands that are an automatic bust (e.g., ['10', '10', '10'])
        hand_value = sum(card_values.get(c, 0) for c in hand)
        if hand.count('A') == 0 and hand_value > 21:
            continue

        for dealer_card in dealer_card_ranks:
            total_tests += 1

            # --- Determine the EXPECTED advice by re-implementing the logic ---
            dealer_val = card_values[dealer_card]
            dealer_index = (dealer_val - 2) if dealer_val <= 10 else 9
            expected_code = ''

            # Calculate hand properties
            num_aces = hand.count('A')
            total_value = sum(card_values.get(c, 0) for c in hand)

            # Priority 1: Pairs (only for 2-card hands)
            is_splittable_pair = (hand_size == 2 and hand[0] == hand[1])
            if is_splittable_pair:
                pair_key = 'A' if hand[0] == 'A' else ('10' if card_values[hand[0]] == 10 else hand[0])
                chart_code = STRATEGY_CHART['pairs'][pair_key][dealer_index]
                if chart_code == 'P':
                    expected_code = 'P'

            # If not a splittable pair, check soft/hard totals
            if not expected_code:
                is_soft = num_aces > 0 and total_value <= 21
                if is_soft:
                    if hand_size == 2 and total_value == 21:  # Blackjack
                        expected_code = 'S'
                    else:
                        soft_total_key = total_value - 11 - (num_aces - 1) * 10
                        if soft_total_key in STRATEGY_CHART['soft']:
                            expected_code = STRATEGY_CHART['soft'][soft_total_key][dealer_index]
                        else:
                            is_soft = False  # Fall through to hard logic if not in chart (e.g., A+A+8=soft 20)

                if not is_soft:  # It's a hard hand
                    hard_total = total_value
                    aces_to_devalue = num_aces
                    while hard_total > 21 and aces_to_devalue > 0:
                        hard_total -= 10
                        aces_to_devalue -= 1

                    if hard_total >= 17:
                        expected_code = 'S'
                    elif hard_total <= 8:
                        expected_code = 'H'
                    elif hard_total in STRATEGY_CHART['hard']:
                        expected_code = STRATEGY_CHART['hard'][hard_total][dealer_index]
                    else:  # Bust hand
                        expected_code = 'S'

            # Convert pure code to playable action
            can_double_or_split = (hand_size == 2)
            if expected_code == 'P':
                expected_advice = "Split"
            elif expected_code == 'D':
                expected_advice = "Double" if can_double_or_split else "Hit"
            elif expected_code == 'Ds':
                expected_advice = "DoubleOrStand" if can_double_or_split else "Stand"
            else:
                expected_advice = action_map.get(expected_code, "Stand")

            # --- Get ACTUAL advice from the function ---
            actual_advice = get_strategy_advice(hand, dealer_card)

            # --- Compare and record failures ---
            if actual_advice != expected_advice:
                failed_tests += 1
                failed_cases.append(
                    f"Hand: {str(hand):<18} vs Dealer: {dealer_card:<2} | "
                    f"Expected: {expected_advice:<13} | Got: {actual_advice}"
                )

    return failed_tests, failed_cases, total_tests


if __name__ == "__main__":
    # --- Test Setup ---
    player_card_ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    dealer_card_ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A']
    card_values = {'A': 11, 'K': 10, 'Q': 10, 'J': 10, '10': 10, '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3,
                   '2': 2}
    action_map = {'H': "Hit", 'S': "Stand", 'D': "Double", 'P': "Split", 'Ds': "DoubleOrStand"}

    # --- Generate Hands ---
    all_2_card_hands = [list(c) for c in combinations_with_replacement(player_card_ranks, 2)]
    all_3_card_hands = [list(c) for c in combinations_with_replacement(player_card_ranks, 3)]
    all_4_card_hands = [list(c) for c in combinations_with_replacement(player_card_ranks, 4)]
    all_5_card_hands = [list(c) for c in combinations_with_replacement(player_card_ranks, 5)]

    # --- Run 2-Card Suite ---
    print("--- Running Comprehensive Strategy Test Suite for 2-Card Hands ---")
    failed_2, cases_2, total_2 = run_test_suite(all_2_card_hands, 2, card_values, dealer_card_ranks, action_map)
    print(f"--- 2-Card Test Complete: {total_2 - failed_2}/{total_2} Passed ---")

    # --- Run 3-Card Suite ---
    print("\n--- Running Comprehensive Strategy Test Suite for 3-Card Hands ---")
    failed_3, cases_3, total_3 = run_test_suite(all_3_card_hands, 3, card_values, dealer_card_ranks, action_map)
    print(f"--- 3-Card Test Complete: {total_3 - failed_3}/{total_3} Passed ---")

    # --- Run 4-Card Suite ---
    print("\n--- Running Comprehensive Strategy Test Suite for 4-Card Hands ---")
    failed_4, cases_4, total_4 = run_test_suite(all_4_card_hands, 4, card_values, dealer_card_ranks, action_map)
    print(f"--- 4-Card Test Complete: {total_4 - failed_4}/{total_4} Passed ---")

    # --- Run 5-Card Suite ---
    print("\n--- Running Comprehensive Strategy Test Suite for 5-Card Hands ---")
    failed_5, cases_5, total_5 = run_test_suite(all_5_card_hands, 5, card_values, dealer_card_ranks, action_map)
    print(f"--- 5-Card Test Complete: {total_5 - failed_5}/{total_5} Passed ---")

    # --- Final Report ---
    total_failed = failed_2 + failed_3 + failed_4 + failed_5
    total_tested = total_2 + total_3 + total_4 + total_5

    print("\n" + "=" * 40)
    print("--- FINAL TEST REPORT ---")
    print(f"Total Scenarios Tested: {total_tested}")
    print(f"Passed: {total_tested - total_failed}")
    print(f"Failed: {total_failed}")
    print("=" * 40)

    if total_failed > 0:
        if cases_2:
            print("\n--- Failed 2-Card Scenarios ---")
            for case in cases_2:
                print(case)
        if cases_3:
            print("\n--- Failed 3-Card Scenarios ---")
            for case in cases_3:
                print(case)
        if cases_4:
            print("\n--- Failed 4-Card Scenarios ---")
            for case in cases_4:
                print(case)
        if cases_5:
            print("\n--- Failed 5-Card Scenarios ---")
            for case in cases_5:
                print(case)
        print("\n❌ Test Suite Failed.")
        sys.exit(1)  # Exit with an error code if tests fail
    else:
        print("\n✅ All tests passed successfully!")