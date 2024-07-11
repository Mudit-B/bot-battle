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
    if len(available) != 0:

        # We will pick the one with the most connections to our territories
        # this should make our territories clustered together a little bit.
        def count_adjacent_friendly(x: int) -> int:
            return len(set(my_territories) & set(game.state.map.get_adjacent_to(x)))

        selected_territory = sorted(available, key=lambda x: count_adjacent_friendly(x), reverse=True)[0]
    
    # Or if there are no such territories, we will pick just an unclaimed one with the greatest degree.
    else:
        # Try not to claim europe and asia coz why try its useless.
        all_territories = set(range(0, 42))

        europe_territories = {9, 10, 11, 12, 13}
        asia_territories = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27}

        # Exclude Europe and Asia territories
        world_except_europe_asia = all_territories - europe_territories - asia_territories
        optimal_choices = world_except_europe_asia & set(unclaimed_territories)
        if optimal_choices:
            selected_territory = random.choice(list(optimal_choices))
        else:
            selected_territory = sorted(unclaimed_territories, key=lambda x: len(game.state.map.get_adjacent_to(x)), reverse=True)[0]
        print(selected_territory, flush=True)

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
    """
    A comprehensive attack function that considers multiple strategic factors.
    """
    my_territories = game.state.get_territories_owned_by(game.state.me.player_id)
    bordering_territories = game.state.get_all_adjacent_territories(my_territories)
    continents = game.state.map.get_continents()
    
    # Update target continent
    bot_state.update_target_continent(game)
    target_continent = bot_state.target_continent

    def calculate_attack_score(attacker: int, target: int) -> float:
        attacker_troops = game.state.territories[attacker].troops
        target_troops = game.state.territories[target].troops
        
        # Base probability of success
        probability = attacker_troops / (target_troops + 1)  # Add 1 to avoid division by zero
        
        # Strategic value of the target territory
        strategic_value = len(game.state.map.get_adjacent_to(target)) / 6  # Normalize by max possible connections
        
        # Continent control factor
        continent_factor = 1
        for continent, territories in continents.items():
            if target in territories:
                owned = len(set(territories) & set(my_territories))
                if owned == len(territories) - 1:  # We're about to complete a continent
                    continent_factor = 3
                    break
                else:
                    continent_factor = 1 + (owned / len(territories))
        
        # Card acquisition factor
        card_factor = 1.5 if len(game.state.me.cards) < 5 else 1
        
        # Troop advantage factor
        troop_advantage = max(0, (attacker_troops - target_troops - 1) / 10)  # Normalize by a reasonable troop difference
        
        # Border security factor
        border_security = 1
        if target in game.state.get_all_border_territories(my_territories):
            border_security = 1.2
        
        # Choke point factor
        choke_point_factor = 1
        target_adjacent = game.state.map.get_adjacent_to(target)
        if len(set(target_adjacent) & set(my_territories)) == 1:
            choke_point_factor = 1.5
        
        # Expansion opportunity factor
        expansion_factor = 1
        new_adjacent = set(game.state.map.get_adjacent_to(target)) - set(my_territories)
        expansion_factor += len(new_adjacent) * 0.1
        
        # Threat reduction factor
        threat_reduction = 1
        enemy_id = game.state.territories[target].occupier
        enemy_territories = game.state.get_territories_owned_by(enemy_id)
        enemy_troops = sum(game.state.territories[t].troops for t in enemy_territories)
        my_troops = sum(game.state.territories[t].troops for t in my_territories)
        if enemy_troops > my_troops * 0.5:  # If enemy is relatively strong
            threat_reduction = 1.3
        
        # Combine all factors
        score = (probability * strategic_value * continent_factor * card_factor * 
                 (1 + troop_advantage) * border_security * choke_point_factor * 
                 expansion_factor * threat_reduction)
        
        # Bonus for target continent
        if any(target in territories for continent, territories in continents.items() if continent == target_continent):
            score *= 1.5
        
        return score

    def find_best_attack() -> Optional[MoveAttack]:
        best_score = 0
        best_move = None

        for target in bordering_territories:
            if game.state.territories[target].occupier == game.state.me.player_id:
                continue  # Skip our own territories
            
            adjacent_territories = set(game.state.map.get_adjacent_to(target))
            potential_attackers = list(adjacent_territories & set(my_territories))
            
            for attacker in potential_attackers:
                attacker_troops = game.state.territories[attacker].troops
                if attacker_troops <= 1:
                    continue  # Need at least 2 troops to attack
                
                score = calculate_attack_score(attacker, target)
                
                if score > best_score:
                    best_score = score
                    attack_troops = min(3, attacker_troops - 1)
                    best_move = game.move_attack(query, attacker, target, attack_troops)

        return best_move

    # Try to find the best attack
    move = find_best_attack()
    if move:
        return move

    # If no good attack is found, pass
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
                    if move_troops > max_troops:
                        max_troops = move_troops
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