from collections import defaultdict, deque
import random
import os
import math
from typing import Optional, Tuple, Union, cast
from risk_helper.game import Game
from risk_shared.models.card_model import CardModel
from risk_shared.queries.query_attack import QueryAttack
from risk_shared.queries.query_claim_territory import QueryClaimTerritory
from risk_shared.queries.query_defend import QueryDefend
from risk_shared.queries.query_distribute_troops import QueryDistributeTroops
from risk_shared.queries.query_fortify import QueryFortify
from risk_shared.queries.query_place_initial_troop import QueryPlaceInitialTroop
from risk_shared.queries.query_redeem_cards import QueryRedeemCards
from risk_shared.queries.query_troops_after_attack import QueryTroopsAfterAttack
from risk_shared.queries.query_type import QueryType
from risk_shared.records.moves.move_attack import MoveAttack
from risk_shared.records.moves.move_attack_pass import MoveAttackPass
from risk_shared.records.moves.move_claim_territory import MoveClaimTerritory
from risk_shared.records.moves.move_defend import MoveDefend
from risk_shared.records.moves.move_distribute_troops import MoveDistributeTroops
from risk_shared.records.moves.move_fortify import MoveFortify
from risk_shared.records.moves.move_fortify_pass import MoveFortifyPass
from risk_shared.records.moves.move_place_initial_troop import MovePlaceInitialTroop
from risk_shared.records.moves.move_redeem_cards import MoveRedeemCards
from risk_shared.records.moves.move_troops_after_attack import MoveTroopsAfterAttack
from risk_shared.records.record_attack import RecordAttack
from risk_shared.records.types.move_type import MoveType



# We will store our enemy in the bot state.
class BotState():
    def __init__(self):
        self.enemy: Optional[int] = None
        self.target_continent = None
        self.conquered_once = False
        
        self.australia = set(range(38, 42))
        self.south_africa = set(range(32, 38))
        self.south_america = set(range(28, 32))

        self.flag_leave_behind_troops = False
        self.flag_continent_capture_leave = False
        self.flag_run = False
        self.killing_final = False

        self.path = []
        self.choke_points = set([29, 30, 2, 0, 4, 10, 13, 14, 15, 36, 34, 33, 22, 16, 26, 21, 24, 40])
        # no adjancey list nonsense for dfs now
        self.map_connections = {0: [1, 5, 21], 1: [6, 5, 0, 8], 2: [3, 8, 30], 3: [7, 6, 8, 2], 4: [5, 6, 7, 10], 5: [4, 0, 1, 6], 6: [7, 4, 5, 1, 8, 3], 7: [4, 6, 3], 8: [3, 6, 1, 2], 9: [11, 12, 10, 15], 10: [12, 4, 9], 11: [14, 12, 9, 15, 13], 12: [14, 10, 9, 11], 13: [22, 14, 11, 15, 36, 34], 14: [16, 26, 12, 11, 13, 22], 15: [13, 11, 9, 36], 16: [17, 26, 14, 22, 18], 17: [23, 25, 26, 16, 18, 24], 18: [24, 17, 16, 22], 19: [21, 27, 25, 23], 20: [21, 23], 21: [0, 27, 19, 23, 20], 22: [18, 16, 14, 13, 34, 33], 23: [20, 21, 19, 25, 17], 24: [17, 18, 40], 25: [27, 26, 17, 23, 19], 26: [25, 14, 16, 17], 27: [21, 25, 19], 28: [29, 31], 29: [36, 30, 31, 28], 30: [2, 31, 29], 31: [29, 30, 28], 32: [33, 36, 37], 33: [22, 34, 36, 32, 37, 35], 34: [22, 13, 36, 33], 35: [33, 37], 36: [33, 34, 13, 15, 29, 32], 37: [35, 33, 32], 38: [39, 41], 39: [40, 41, 38], 40: [39, 24, 41], 41: [38, 39, 40]}

        self.mark_only_move_three = False

    def update_target_continent(self, game: Game):
        """Update the target continent based on the current game state."""
        continents = game.state.map.get_continents()
        my_territories = game.state.get_territories_owned_by(game.state.me.player_id)

        # Calculate the percentage of each continent controlled by us
        continent_control = {}
        for continent, territories in continents.items():
            owned = len(set(territories) & set(my_territories))
            total = len(territories)
            continent_control[continent] = (owned / total, total - owned)

        for continent in continent_control:
            if continent_control[continent][0] == 1:
                # Don't choose this continent if it's already fully controlled
                continent_control[continent] = (-1, -1)

        # Determine the continent closest to being captured by us
        self.target_continent = max(continent_control, key=lambda x: (continent_control[x][0], -continent_control[x][1]))

    # returns the path we need.
    def dfs_path(self, vertices, start) -> list[int]:
        # subproblem
        results = []
        def dfs(current, path, remaining) -> list[int]:
            # path and remaining are different
            path.append(current)

            # There is a case where 1 -> 2 -> 3
            # visit 2
            # path [2]
            # remaining {1, 3}
            if current in remaining:
                remaining.remove(current)
            
            # Return the curr path
            if not remaining:
                return path
            
            for neighbor in self.map_connections[current]:
                if neighbor in remaining:
                    result = dfs(neighbor, path.copy(), remaining.copy())
                    results.append(result)

                    if result:
                        return result
            return []
        
        remaining = set(vertices) - {start}
        result = dfs(start, [], remaining)
        if results:
            return max(results, key=len)
        return []
    
    def get_num_alive(self, game):
        return len([i for i in game.state.players if game.state.players[i].alive])
    
def main():
    
    # Get the game object, which will connect you to the engine and
    # track the state of the game.
    game = Game()
    bot_state = BotState()

    # Respond to the engine's queries with your moves.
    while True:

        # Get the engine's query (this will block until you receive a query).
        query = game.get_next_query()

        # Based on the type of query, respond with the correct move.
        def choose_move(query: QueryType) -> MoveType:
            match query:
                case QueryClaimTerritory() as q:
                    return handle_claim_territory(game, bot_state, q)

                case QueryPlaceInitialTroop() as q:
                    return handle_place_initial_troop(game, bot_state, q)

                case QueryRedeemCards() as q:
                    return handle_redeem_cards(game, bot_state, q)

                case QueryDistributeTroops() as q:
                    return handle_distribute_troops(game, bot_state, q)

                case QueryAttack() as q:
                    return handle_attack(game, bot_state, q)

                case QueryTroopsAfterAttack() as q:
                    return handle_troops_after_attack(game, bot_state, q)

                case QueryDefend() as q:
                    return handle_defend(game, bot_state, q)

                case QueryFortify() as q:
                    return handle_fortify(game, bot_state, q)
        
        # Send the move to the engine.
        game.send_move(choose_move(query))

def handle_claim_territory(game: Game, bot_state: BotState, query: QueryClaimTerritory) -> MoveClaimTerritory:
    """At the start of the game, you can claim a single unclaimed territory every turn 
    until all the territories have been claimed by players."""

    unclaimed_territories = game.state.get_territories_owned_by(None)
    my_territories = game.state.get_territories_owned_by(game.state.me.player_id)

    # We will try to always pick new territories that are next to ones that we own,
    # or a random one if that isn't possible.
    adjacent_territories = game.state.get_all_adjacent_territories(my_territories)

    # We can only pick from territories that are unclaimed and adjacent to us.
    available = list(set(unclaimed_territories) & set(adjacent_territories))

    left_australia = set(unclaimed_territories) & set(bot_state.australia)
    left_south_africa = set(unclaimed_territories) & set(bot_state.south_africa)
    left_south_america = set(unclaimed_territories) & set(bot_state.south_america)

    if len(available) != 0:
        if left_australia:
            selected_territory = random.choice(list(left_australia))
        elif left_south_america:
            selected_territory = random.choice(list(left_south_america))
        elif left_south_africa:
            selected_territory = random.choice(list(left_south_africa))
        else:
            # We will pick the one with the most connections to our territories
            # this should make our territories clustered together a little bit.
            def count_adjacent_friendly(x: int) -> int:
                return len(set(my_territories) & set(game.state.map.get_adjacent_to(x)))

            selected_territory = sorted(available, key=lambda x: count_adjacent_friendly(x), reverse=True)[0]
    # Or if there are no such territories, we will pick just an unclaimed one with the greatest degree.
    else:
        if left_australia:
            selected_territory = random.choice(list(left_australia))
        elif left_south_america:
            selected_territory = random.choice(list(left_south_america))
        elif left_south_africa:
            selected_territory = random.choice(list(left_south_africa))
        else:
            selected_territory = sorted(unclaimed_territories, key=lambda x: len(game.state.map.get_adjacent_to(x)), reverse=True)[0]

    return game.move_claim_territory(query, selected_territory)

def handle_place_initial_troop(game: Game, bot_state: BotState, query: QueryPlaceInitialTroop) -> MovePlaceInitialTroop:
    """After all the territories have been claimed, you can place a single troop on one
    of your territories each turn until each player runs out of troops."""
    
    # We will place troops along the territories on our border.
    border_territories = game.state.get_all_border_territories(
        game.state.get_territories_owned_by(game.state.me.player_id)
    )

   # We will distribute troops across our border territories.
    total_troops = game.state.me.troops_remaining
    distributions = defaultdict(lambda: 0)
    border_territories = game.state.get_all_border_territories(
        game.state.get_territories_owned_by(game.state.me.player_id)
    )

    # We need to remember we have to place our matching territory bonus
    # if we have one.
    if len(game.state.me.must_place_territory_bonus) != 0:
        assert total_troops >= 2
        distributions[game.state.me.must_place_territory_bonus[0]] += 2
        total_troops -= 2

    # Prioritize capturing whole continents
    continents = game.state.map.get_continents()
    my_territories = game.state.get_territories_owned_by(game.state.me.player_id)

    # this gets the target continent.
    bot_state.update_target_continent(game)
    target_continent = bot_state.target_continent

    def eval_terr(territory):
        # like chess we eval the val of this territory
        value = 0
        continent = None
        for k, territories in continents.items():
            if territory in territories:
                continent = k
                break
        if continent == target_continent:
            value += 100
        # 1% this too hit and trial this.
        value += 0.5 * len(set(game.state.map.get_adjacent_to(territory)) & set(my_territories))
        return value

    def find_best_territory(lst):
        selected_territory = None
        for territory in lst:
            if selected_territory is None:
                selected_territory = territory
            else:
                if eval_terr(territory) > eval_terr(selected_territory):
                    selected_territory = territory
        return selected_territory

    # Find the best border territory to distribute troops
    best_border_territory = find_best_territory(border_territories)

    if best_border_territory:
        distributions[best_border_territory] += total_troops
    else:
        # If no specific border territory is found, distribute the maximum number of troops to the territory with the highest attack potential
        max_attack_territory = max(border_territories, key=lambda x: game.state.territories[x].troops)
        distributions[max_attack_territory] += total_troops

    # return game.move_distribute_troops(query, distributions)
    if best_border_territory is not None:
        return game.move_place_initial_troop(query, best_border_territory)
    else:
        # We will place troops along the territories on our border.
        border_territories = game.state.get_all_border_territories(
            game.state.get_territories_owned_by(game.state.me.player_id)
        )

        # We will place a troop in the border territory with the least troops currently
        # on it. This should give us close to an equal distribution.
        border_territory_models = [game.state.territories[x] for x in border_territories]
        min_troops_territory = min(border_territory_models, key=lambda x: x.troops)

        return game.move_place_initial_troop(query, min_troops_territory.territory_id)


def handle_redeem_cards(game: Game, bot_state: BotState, query: QueryRedeemCards) -> MoveRedeemCards:
    """After the claiming and placing initial troops phases are over, you can redeem any
    cards you have at the start of each turn, or after killing another player."""

    # We will always redeem the minimum number of card sets we can until the 12th card set has been redeemed.
    # This is just an arbitrary choice to try and save our cards for the late game.

    # We always have to redeem enough cards to reduce our card count below five.
    card_sets: list[Tuple[CardModel, CardModel, CardModel]] = []
    cards_remaining = game.state.me.cards.copy()

    while len(cards_remaining) >= 5:
        card_set = game.state.get_card_set(cards_remaining)
        # According to the pigeonhole principle, we should always be able to make a set
        # of cards if we have at least 5 cards.
        assert card_set != None
        card_sets.append(card_set)
        cards_remaining = [card for card in cards_remaining if card not in card_set]

    # Remember we can't redeem any more than the required number of card sets if 
    # we have just eliminated a player.
    if game.state.card_sets_redeemed > 12 and query.cause == "turn_started":
        card_set = game.state.get_card_set(cards_remaining)
        while card_set != None:
            card_sets.append(card_set)
            cards_remaining = [card for card in cards_remaining if card not in card_set]
            card_set = game.state.get_card_set(cards_remaining)

    return game.move_redeem_cards(query, [(x[0].card_id, x[1].card_id, x[2].card_id) for x in card_sets])

def handle_distribute_troops(game: Game, bot_state: BotState, query: QueryDistributeTroops) -> MoveDistributeTroops:
    """After you redeem cards (you may have chosen to not redeem any), you need to distribute
    all the troops you have available across your territories. This can happen at the start of
    your turn or after killing another player.
    """

    # We will distribute troops across our border territories.
    total_troops = game.state.me.troops_remaining
    distributions = defaultdict(lambda: 0)
    border_territories = game.state.get_all_border_territories(
        game.state.get_territories_owned_by(game.state.me.player_id)
    )

    # We need to remember we have to place our matching territory bonus
    # if we have one.
    if len(game.state.me.must_place_territory_bonus) != 0:
        assert total_troops >= 2
        distributions[game.state.me.must_place_territory_bonus[0]] += 2
        total_troops -= 2

    # Prioritize capturing whole continents
    continents = game.state.map.get_continents()
    my_territories = game.state.get_territories_owned_by(game.state.me.player_id)

    # this gets the target continent.
    bot_state.update_target_continent(game)
    target_continent = bot_state.target_continent

    def eval_terr(territory):
        # like chess we eval the val of this territory
        value = 0
        continent = None
        for k, territories in continents.items():
            if territory in territories:
                continent = k
                break
        if continent == target_continent:
            value += 100
        # 1% this too hit and trial this.
        value += 0.5 * len(set(game.state.map.get_adjacent_to(territory)) & set(my_territories))
        return value

    def find_best_territory(lst):
        selected_territory = None
        for territory in lst:
            if selected_territory is None:
                selected_territory = territory
            else:
                if eval_terr(territory) > eval_terr(selected_territory):
                    selected_territory = territory
        return selected_territory

    # Find the best border territory to distribute troops
    best_border_territory = find_best_territory(border_territories)

    if best_border_territory is not None:
        if bot_state.get_num_alive(game) == 2:
            get_ones = [t for t in my_territories if game.state.territories[t].troops == 1]
            if len(get_ones) < total_troops:
                for ones in get_ones:
                    distributions[ones] += 1
                    total_troops -= 1
        distributions[best_border_territory] += total_troops
    else:
        # If no specific border territory is found, distribute the maximum number of troops to the territory with the highest attack potential
        max_attack_territory = max(border_territories, key=lambda x: game.state.territories[x].troops)
        distributions[max_attack_territory] += total_troops

    return game.move_distribute_troops(query, distributions)

def handle_attack(game: Game, bot_state: BotState, query: QueryAttack) -> Union[MoveAttack, MoveAttackPass]:
    """After the troop phase of your turn, you may attack any number of times until you decide to
    stop attacking (by passing). After a successful attack, you may move troops into the conquered
    territory. If you eliminated a player you will get a move to redeem cards and then distribute troops."""
    
    save_move_for_bot_path = [0, 0]
    bot_state.flag_leave_behind_troops = False
    bot_state.flag_continent_capture_leave = False
    bot_state.killing_final = False

    continents = game.state.map.get_continents()

    def get_continent_of_territory(territory):
        for k, v in continents.items():
            if territory in v:
                return k
        return -1
    def get_territories_of_continent_of_territory(territory):
        for k, v in continents.items():
            if territory in v:
                return v
        return []
    
    def is_positive_move(attacker, target):
        # attacker surronding
        attacker_troops = game.state.territories[attacker].troops
        target_troops = game.state.territories[target].troops
        my_territories = game.state.get_territories_owned_by(game.state.me.player_id)
        adj_attacker = set(game.state.map.get_adjacent_to(attacker)) - set(my_territories)

        attacker_continent = bot_state.target_continent
        sum_continent_enemy = 0
        my_score = 0
        for k, v in game.state.map.get_continents().items():
            if target in v:
                # always reaches here so shouldnt be an issue
                attacker_continent = k
                for t in v:
                    if t not in my_territories:
                        sum_continent_enemy += game.state.territories[t].troops
                    else:
                        # penalty just for 1's and should avoid over extending.
                        my_score += game.state.territories[t].troops - 1
                # add choke points
                continent_terrs = set(v)
                choke_pts = bot_state.choke_points & continent_terrs
                mine = set(my_territories)
                for choke_point in choke_pts:
                    bordering = game.state.map.get_adjacent_to(choke_point)
                    other_enemies = set(bordering) - mine - continent_terrs
                    sum_continent_enemy += max(other_enemies, key=lambda t: game.state.territories[t].troops, default=0)
                break

        if get_continent_of_territory(attacker) != get_continent_of_territory(target):
            my_score += attacker_troops
        # print(game.state.recording, my_score, self.attacker, target)
        if sum_continent_enemy < my_score:
            return True
        return False

    def attack_highest_probability(territories: list[int]):

        best_probability = 0
        best_move = None
        actual_best = 0
        must_make = None

        for target in territories:
            # Find my attackers
            adjacent_territories = set(game.state.map.get_adjacent_to(target))
            potential_attackers = list(adjacent_territories & set(my_territories))
            
            # No attackers for this target, check the next target
            if not potential_attackers:
                continue

            potential_attackers = sorted(potential_attackers, key=lambda t: game.state.territories[t].troops, reverse=True)
            # Find the highest attacker from my attackers
            for attacker in potential_attackers:

                attacker_troops = game.state.territories[attacker].troops
                target_troops = game.state.territories[target].troops

                # Calculate the probability of success
                probability = attacker_troops / target_troops if target_troops > 0 else float('inf')

                # Determine if this is a favorable attack
                is_favorable_attack = (attacker_troops - target_troops >= 2) and is_positive_move(attacker, target)
                if is_favorable_attack and probability > best_probability:
                    save_move_for_bot_path[0] = attacker
                    save_move_for_bot_path[1] = target
                    best_probability = probability
                    best_move = game.move_attack(query, attacker, target, min(3, attacker_troops - 1))
                elif (attacker_troops - target_troops >= 2) and probability > actual_best:
                    actual_best = probability
                    must_make = game.move_attack(query, attacker, target, min(3, attacker_troops - 1))
         
        if not best_move and (bot_state.conquered_once == False):
            return must_make, False
        return best_move, True
    
    def eliminate_final_enemy(game: Game, bot_state: BotState, query: QueryAttack):
        my_territories = game.state.get_territories_owned_by(game.state.me.player_id)
        if bot_state.path != []:
            while bot_state.path and bot_state.path[0] in my_territories:
                bot_state.path.pop(0)
            if bot_state.path != []:
                target = bot_state.path[0]
                potential_attackers = set(game.state.map.get_adjacent_to(target)) & set(my_territories)
                potential_chain_enemies = set(game.state.map.get_adjacent_to(target)) - set(my_territories)

                potential_chain_enemies.union(set(game.state.get_all_adjacent_territories(list(potential_attackers))) - set(my_territories))

                for enemy in potential_chain_enemies:
                    if enemy not in bot_state.path:
                        print("HERE", flush=True)
                        bot_state.flag_leave_behind_troops = True

                if potential_attackers != set():
                    attacker = max(potential_attackers, key=lambda x: game.state.territories[x].troops)
                    attacker_troops = game.state.territories[attacker].troops
                    target_troops = game.state.territories[target].troops

                    is_favorable_attack = (attacker_troops - target_troops >= 2) and is_positive_move(attacker, target)

                    if is_favorable_attack:
                        return game.move_attack(query, attacker, target, min(3, attacker_troops - 1))
        else:
            if bot_state.enemy is None:
                last_player = None
                for enemy in game.state.players:
                    if game.state.players[enemy].alive:
                        last_player = game.state.players[enemy].player_id
                bot_state.enemy = last_player

            
            # This should automatically get the enemy killed.
            move, isContinentCapture = attack_highest_probability(bordering_territories)
            if move:
                if isContinentCapture:
                    attacker, defender = save_move_for_bot_path
                    vertices_to_capture = set(game.state.get_territories_owned_by(bot_state.enemy))
                    bot_state.path = bot_state.dfs_path(vertices_to_capture, defender)
                    # print(len(game.state.recording), attacker, defender, vertices_to_capture, bot_state.path, flush=True)
                return move
    # We will attack someone.
    my_territories = game.state.get_territories_owned_by(game.state.me.player_id)
    bordering_territories = game.state.get_all_adjacent_territories(my_territories)
    
    if bot_state.get_num_alive(game) == 2:
        if bot_state.enemy is None:
            last_player = None
            for enemy in game.state.players:
                if game.state.players[enemy].alive:
                    last_player = game.state.players[enemy].player_id
            bot_state.enemy = last_player
        last_player = bot_state.enemy
        
        total_enemy_troops = sum([game.state.territories[t].troops for t in game.state.get_territories_owned_by(last_player)])
        total_my_score = 0
        for my_t in bordering_territories:
            total_my_score += game.state.territories[my_t].troops - 1
        if total_enemy_troops <= my_t:
            res =  eliminate_final_enemy(game, bot_state, query)
            if res:
                bot_state.killing_final = True
                return res

    
    # update target continent
    bot_state.update_target_continent(game)
    target_continent = bot_state.target_continent

    
    # continent capture leave.
    if bot_state.path != []:
        while bot_state.path and bot_state.path[0] in set(my_territories):
            bot_state.path.pop(0)
        if bot_state.path != []:
            target = bot_state.path[0]
            continent = get_continent_of_territory(target)
            other_targets = set(get_territories_of_continent_of_territory(target)) - set(my_territories)

            for other_target in other_targets:
                if get_continent_of_territory(other_target) == continent:
                    if other_target not in bot_state.path:
                        bot_state.flag_continent_capture_leave = True

            potential_attackers = set(game.state.map.get_adjacent_to(target)) & set(my_territories)
            if potential_attackers != set():
                attacker = max(potential_attackers, key=lambda x: game.state.territories[x].troops)

                attacker_troops = game.state.territories[attacker].troops
                target_troops = game.state.territories[target].troops

                is_favorable_attack = (attacker_troops - target_troops >= 2) and is_positive_move(attacker, target)

                if is_favorable_attack:
                    return game.move_attack(query, attacker, target, min(3, attacker_troops - 1))

    bot_state.path = []


    def find_best_target_continent(territories: list[int]) -> list[int]:
        # find territories to those in the target continent
        target_territories = []
        for territory in territories:
            for continent, continent_territories in continents.items():
                if continent == target_continent and territory in continent_territories:
                    target_territories.append(territory)
        return target_territories

    target_territories = find_best_target_continent(bordering_territories)

    # We will attack the target territory with the highest probability of success if possible.
    move, isContinentCapture = attack_highest_probability(target_territories)
    if move:
        if isContinentCapture:
            attacker, defender = save_move_for_bot_path
            continet_to_be_captured = get_continent_of_territory(defender)
            vertices_to_capture = set(game.state.map.get_continents()[continet_to_be_captured]) - set(my_territories)
            bot_state.path = bot_state.dfs_path(vertices_to_capture, defender)
            # print(len(game.state.recording), attacker, defender, vertices_to_capture, bot_state.path, flush=True)
        elif bot_state.conquered_once:
            return game.move_attack_pass(query)
        return move

    # Otherwise, attack any bordering territory with the highest probability of success.
    move, isContinentCapture = attack_highest_probability(bordering_territories)
    if move:
        if isContinentCapture:
            attacker, defender = save_move_for_bot_path
            continet_to_be_captured = get_continent_of_territory(defender)
            vertices_to_capture = set(game.state.map.get_continents()[continet_to_be_captured]) - set(my_territories)
            bot_state.path = bot_state.dfs_path(vertices_to_capture, defender)
            # print(len(game.state.recording), attacker, defender, vertices_to_capture, bot_state.path, flush=True)
        elif bot_state.conquered_once:
            return game.move_attack_pass(query)
        return move
    
    # try last check if there is only one adjacent enemy to an attacker
    best_probablity = 0
    border_territories = game.state.get_all_border_territories(my_territories)
    for t in border_territories:
        enemies = set(game.state.map.get_adjacent_to(t)) - set(my_territories)
        if len(enemies) == 1:
            enemy = list(enemies)[0]
            attacker_troops = game.state.territories[t].troops
            target_troops = game.state.territories[enemy].troops
            probability = attacker_troops / target_troops if target_troops > 0 else float('inf')
            if (attacker_troops - target_troops >= 2) and (probability > best_probablity):
                best_probablity = probability
                # print(t, enemy, attacker_troops, target_troops)
                move = game.move_attack(query, t, enemy, min(3, attacker_troops - 1))
    if move:
        # print(move, flush=True)
        return move
    return game.move_attack_pass(query)
    
def handle_troops_after_attack(game: Game, bot_state: BotState, query: QueryTroopsAfterAttack) -> MoveTroopsAfterAttack:
    """After conquering a territory in an attack, you must move troops to the new territory."""
    bot_state.conquered_once = True
    
    # First we need to get the record that describes the attack, and then the move that specifies
    # which territory was the attacking territory.
    record_attack = cast(RecordAttack, game.state.recording[query.record_attack_id])
    move_attack = cast(MoveAttack, game.state.recording[record_attack.move_attack_id])
    
    target = move_attack.defending_territory
    attacker = move_attack.attacking_territory
    
    adj_to_target =  game.state.map.get_adjacent_to(target)
    mine = game.state.get_territories_owned_by(game.state.me.player_id)
    
    enemy_adj_to_target = set(adj_to_target) - set(mine)
    curr_adj_to_attacker = set(game.state.map.get_adjacent_to(attacker)) - set(mine)
    
    if (enemy_adj_to_target == set()):
        # If there is no enemy then move only 3
        move_troops = min(3, game.state.territories[move_attack.attacking_territory].troops - 1)
        return game.move_troops_after_attack(query, move_troops)

    print(len(game.state.recording), bot_state.path, 'HERE in handle troops after attack', flush=True)

    # If the target has some new attacker
    if (curr_adj_to_attacker <= (enemy_adj_to_target)) or (bot_state.path != []):
        # We will always move the maximum number of troops we can.
        if bot_state.get_num_alive(game) == 2:
            if game.state.territories[move_attack.attacking_territory].troops > 10:
                return game.move_troops_after_attack(query, game.state.territories[move_attack.attacking_territory].troops - 2)
        return game.move_troops_after_attack(query, game.state.territories[move_attack.attacking_territory].troops - 1)
    else:
        # If there is no enemy then move only 3
        move_troops = min(3, game.state.territories[move_attack.attacking_territory].troops - 1)
        return game.move_troops_after_attack(query, move_troops)

def handle_defend(game: Game, bot_state: BotState, query: QueryDefend) -> MoveDefend:
    """If you are being attacked by another player, you must choose how many troops to defend with."""

    # We will always defend with the most troops that we can.

    # First we need to get the record that describes the attack we are defending against.
    move_attack = cast(MoveAttack, game.state.recording[query.move_attack_id])
    defending_territory = move_attack.defending_territory
    
    # We can only defend with up to 2 troops, and no more than we have stationed on the defending
    # territory.
    defending_troops = min(game.state.territories[defending_territory].troops, 2)
    return game.move_defend(query, defending_troops)


def handle_fortify(game: Game, bot_state: BotState, query: QueryFortify) -> Union[MoveFortify, MoveFortifyPass]:
    """At the end of your turn, after you have finished attacking, you may move a number of troops between
    any two of your territories (they must be adjacent)."""
    
    bot_state.conquered_once = False
    """
    Previous function was really suboptimal i think?
    Idea is to find max threat
    This implementation tries to find the bordering territories 
    """
    my_territories = game.state.get_territories_owned_by(game.state.me.player_id)
    border_territories = game.state.get_all_border_territories(my_territories)

    # max troops to move to the best move
    max_troops = 0

    """
    best move is the greedy move in which we move from 
    highest amounnt of troops to lowest bordering amount of troops territory
    """
    # tuple used to store best move state to be extracted later
    best_move = None
    max_fortification_value = 0
    mine = set(my_territories)

    def get_threat(territory):
        return sum(game.state.territories[t].troops for t in set(game.state.map.get_adjacent_to(territory)) - mine)

    candidate_territories = game.state.get_all_border_territories(my_territories)
    candidate_threat = []
    for territory in candidate_territories:
        threat = get_threat(territory)
        candidate_threat.append((threat, territory))
    candidate_threat.sort(reverse=True)

    most_troops_territory = sorted(my_territories, key=lambda x: game.state.territories[x].troops, reverse=True)
    for threat_level, target_territory in candidate_threat: 
        for territory in most_troops_territory:
            if territory == target_territory:
                continue
            if target_territory not in game.state.map.get_adjacent_to(territory):
                continue
            
            source_troops = game.state.territories[territory].troops
            target_troops = game.state.territories[target_territory].troops
            
            if source_troops <= 1:
                continue
            
            threat = get_threat(territory)
            
            if threat_level <= threat:
                continue
            total_threat = threat_level + threat
            if total_threat <= 0:
                continue
            fraction_to_send = threat_level / total_threat

            # the optimal number of troops to move
            troops = math.floor((game.state.territories[territory].troops - 1) * fraction_to_send)
            troops = max(troops, 1)

            # the fortification value => improvement in the position
            fortification_value = (threat_level - threat) * troops
            
            if fortification_value > max_fortification_value:
                max_fortification_value = fortification_value
                best_move = (territory, target_territory, troops)

    if best_move:
        return game.move_fortify(query, best_move[0], best_move[1], best_move[2])

    return game.move_fortify_pass(query)

if __name__ == "__main__":
    main()