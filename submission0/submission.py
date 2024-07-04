from collections import defaultdict, deque
import random
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

        # Log the target continent for debugging purposes
        print(f"Updated target continent: {self.target_continent}", flush=True)
    
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
    adjacent_territories = game.state.get_all_adjacent_territories(my_territories)

    available = list(set(unclaimed_territories) & set(adjacent_territories))

    '''
    Idea is to claim the territories that are closer to the capturing a whole island / continent
    '''
    # continents closest to getting captured by us
    continents = game.state.map.get_continents()
    
    # this gets the target continent.
    bot_state.update_target_continent(game)
    target_continent = bot_state.target_continent
    
    def eval_terr(territory):
        # like chess we eval the val of this territory
        value = 0
        continent = None
        for k, territories in continents.items():
            if territory in territories:
                continent = continents[k]
                break
        if continent == target_continent:
            value += 3
        print(territory, continent, target_continent, flush=True)
        # 1% this too hit and trial this.
        value += 0.5 * len(set(game.state.map.get_adjacent_to(territory)) & set(my_territories))
        return value

    def find_best_territory(lst):
        selected_territory = None
        for territory in lst:
            if selected_territory == None:
                selected_territory = territory
            else:
                if eval_terr(territory) > eval_terr(selected_territory):
                    selected_territory = territory
        return selected_territory

    selected_territory = None
    if available:
        selected_territory = find_best_territory(available)
    if not selected_territory:
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

    if best_border_territory:
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
                is_favorable_attack = (attacker_troops - target_troops >= 2) and (attacker_troops > 4)
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

    if len(game.state.recording) < 4000:
        # get the territories in the target continent
        target_territories = find_best_target_continent(bordering_territories)

        # We will attack the target territory with the highest probability of success if possible.
        move = attack_highest_probability(target_territories)
        if move:
            return move

        # Otherwise, attack any bordering territory with the highest probability of success.
        move = attack_highest_probability(bordering_territories)
        if move:
            return move

    # In the late game, attack anyone adjacent to our strongest territories (hopefully our doomstack).
    else:
        strongest_territories = sorted(my_territories, key=lambda x: game.state.territories[x].troops, reverse=True)
        for territory in strongest_territories:
            move = attack_highest_probability(list(set(game.state.map.get_adjacent_to(territory)) - set(my_territories)))
            if move:
                return move

    return game.move_attack_pass(query)


def handle_troops_after_attack(game: Game, bot_state: BotState, query: QueryTroopsAfterAttack) -> MoveTroopsAfterAttack:
    """After conquering a territory in an attack, you must move troops to the new territory."""
    
    # First we need to get the record that describes the attack, and then the move that specifies
    # which territory was the attacking territory.
    record_attack = cast(RecordAttack, game.state.recording[query.record_attack_id])
    move_attack = cast(MoveAttack, game.state.recording[record_attack.move_attack_id])

    # We will always move the maximum number of troops we can.
    return game.move_troops_after_attack(query, game.state.territories[move_attack.attacking_territory].troops - 1)


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


    # We will always fortify towards the most powerful player (player with most troops on the map) to defend against them.
    my_territories = game.state.get_territories_owned_by(game.state.me.player_id)
    total_troops_per_player = {}
    for player in game.state.players.values():
        total_troops_per_player[player.player_id] = sum([game.state.territories[x].troops for x in game.state.get_territories_owned_by(player.player_id)])

    most_powerful_player = max(total_troops_per_player.items(), key=lambda x: x[1])[0]

    # If we are the most powerful, we will pass.
    if most_powerful_player == game.state.me.player_id:
        return game.move_fortify_pass(query)
    
    # Otherwise we will find the shortest path between our territory with the most troops
    # and any of the most powerful player's territories and fortify along that path.
    candidate_territories = game.state.get_all_border_territories(my_territories)
    most_troops_territory = max(candidate_territories, key=lambda x: game.state.territories[x].troops)

    # To find the shortest path, we will use a custom function.
    shortest_path = find_shortest_path_from_vertex_to_set(game, most_troops_territory, set(game.state.get_territories_owned_by(most_powerful_player)))
    # We will move our troops along this path (we can only move one step, and we have to leave one troop behind).
    # We have to check that we can move any troops though, if we can't then we will pass our turn.
    if len(shortest_path) > 0 and game.state.territories[most_troops_territory].troops > 1:
        return game.move_fortify(query, shortest_path[0], shortest_path[1], game.state.territories[most_troops_territory].troops - 1)
    else:
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