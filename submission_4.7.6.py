from collections import defaultdict, deque
import random
import os
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
        
        self.australia = set(range(38, 42))
        self.south_africa = set(range(32, 38))
        self.south_america = set(range(28, 32))

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
    my_territories = set(game.state.get_territories_owned_by(game.state.me.player_id))

    # We will try to always pick new territories that are next to ones that we own,
    # or a random one if that isn't possible.
    adjacent_territories = game.state.get_all_adjacent_territories(list(my_territories))

    # We can only pick from territories that are unclaimed and adjacent to us.
    available = list(set(unclaimed_territories) & set(adjacent_territories))

    left_australia = set(unclaimed_territories) & set(bot_state.australia)
    if left_australia != bot_state.australia - my_territories:
        left_australia = set()
    
    left_south_africa = set(unclaimed_territories) & set(bot_state.south_africa)
    if left_south_africa != bot_state.south_africa - my_territories:
        left_south_africa = set()

    left_south_america = set(unclaimed_territories) & set(bot_state.south_america)
    if left_south_america != bot_state.south_america - my_territories:
        left_south_america = set()

    if len(available) != 0:
        if left_australia:
            selected_territory = random.choice(list(left_australia))
        elif left_south_africa:
            selected_territory = random.choice(list(left_south_africa))
        elif left_south_america:
            selected_territory = random.choice(list(left_south_america))
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
        elif left_south_africa:
            selected_territory = random.choice(list(left_south_africa))
        elif left_south_america:
            selected_territory = random.choice(list(left_south_america))
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

    # We will attack someone.
    my_territories = game.state.get_territories_owned_by(game.state.me.player_id)
    bordering_territories = game.state.get_all_adjacent_territories(my_territories)
    continents = game.state.map.get_continents()
    
    # update target continent
    bot_state.update_target_continent(game)
    target_continent = bot_state.target_continent

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
                        my_score += game.state.territories[t].troops

                if attacker in v:
                    my_score -= attacker_troops
                break
        
        my_score += attacker_troops
        print(len(game.state.recording), f'From: {game.state.map._vertex_names[attacker]} to: {game.state.map._vertex_names[target]}')
        print(len(game.state.recording), f"My Score: {my_score}, sum enemy: {sum_continent_enemy}")
        if sum_continent_enemy < my_score:
            return True
        return False



    def attack_highest_probability(territories: list[int]) -> Optional[MoveAttack]:
        best_probability = 0
        best_move = None

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
                    best_probability = probability
                    best_move = game.move_attack(query, attacker, target, min(3, attacker_troops - 1))

        return best_move

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
    move = attack_highest_probability(target_territories)
    if move:
        return move

    # Otherwise, attack any bordering territory with the highest probability of success.
    move = attack_highest_probability(bordering_territories)
    if move:
        return move

    return game.move_attack_pass(query)

def handle_troops_after_attack(game: Game, bot_state: BotState, query: QueryTroopsAfterAttack) -> MoveTroopsAfterAttack:
    """After conquering a territory in an attack, you must move troops to the new territory."""
    
    # First we need to get the record that describes the attack, and then the move that specifies
    # which territory was the attacking territory.
    record_attack = cast(RecordAttack, game.state.recording[query.record_attack_id])
    move_attack = cast(MoveAttack, game.state.recording[record_attack.move_attack_id])
    
    defender = move_attack.defending_territory
    attacker = move_attack.attacking_territory
    
    adjacent_to_def =  game.state.map.get_adjacent_to(defender)
    mine = game.state.get_territories_owned_by(game.state.me.player_id)
    bordering = set(adjacent_to_def) - set(mine)
    if bordering:
        # We will always move the maximum number of troops we can.
        return game.move_troops_after_attack(query, game.state.territories[move_attack.attacking_territory].troops - 1)
    else:
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
    max_move = 0

    # check all our territories for the optimal strat
    for territory in my_territories:

        # Cant really move if 1 troop
        if game.state.territories[territory].troops <= 1:
            continue
        
        # Check the neighbors of the current territory
        for neighbor in game.state.map.get_adjacent_to(territory):

            # obv we can only fortify our territories
            if neighbor in my_territories:
                if game.state.territories[neighbor].troops < max_troops:
                    continue

                # if neighbor is a border territory
                if neighbor in border_territories:
                    # calc the troops that can be moved, we move fortify max 
                    move_troops = game.state.territories[territory].troops - 1

                    from_enemies = set(game.state.map.get_adjacent_to(territory)) - set(my_territories)
                    to_enemies = set(game.state.map.get_adjacent_to(neighbor)) - set(my_territories)

                    from_enemies_count = sum([game.state.territories[i].troops for i in from_enemies])
                    to_enemies_count = sum([game.state.territories[i].troops for i in to_enemies])
                    if max_move < (to_enemies_count - from_enemies_count):
                        max_move = to_enemies_count - from_enemies_count
                        best_move = (territory, neighbor, move_troops)

    # execute the best move if we have one
    if best_move:
        from_territory, to_territory, troops = best_move

        return game.move_fortify(query, from_territory, to_territory, troops)
    else:
        # otherwise, pass the fortify move
        return game.move_fortify_pass(query)



def find_shortest_path_from_vertex_to_set(game: Game, source: int, target_set: set[int]) -> list[int]:
    """Used in move_fortify()."""

    # We perform a BFS search from our source vertex, stopping at the first member of the target_set we find.
    queue = deque()
    queue.appendleft(source)

    current = queue.pop()
    parent = {}
    seen = {current: True}

    while len(queue) != 0:
        if current in target_set:
            break

        for neighbour in game.state.map.get_adjacent_to(current):
            if neighbour not in seen:
                seen[neighbour] = True
                parent[neighbour] = current
                queue.appendleft(neighbour)

        current = queue.pop()

    path = []
    while current in parent:
        path.append(current)
        current = parent[current]

    return path[::-1]

if __name__ == "__main__":
    main()