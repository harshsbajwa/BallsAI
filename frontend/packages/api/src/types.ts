// Generic type for paginated responses
export interface PaginatedResponse<T> {
  total: number;
  limit: number;
  offset: number;
  items: T[];
}

// Base Models
export interface Player {
  person_id: number;
  full_name: string;
  primary_position?: string;
  height?: number;
  body_weight?: number;
  birthdate?: string;
  country?: string;
  draft_year?: number;
  draft_round?: number;
  draft_number?: number;
}

export interface Team {
  team_id: number;
  full_name: string;
  team_abbrev?: string;
  team_city: string;
  team_name: string;
}

export interface Game {
  game_id: number;
  game_date: string;
  home_team: Team;
  away_team: Team;
  home_score?: number;
  away_score?: number;
  winning_team_id?: number;
}

export interface PlayerStats {
  game_id: number;
  game_date: string;
  points: number;
  assists: number;
  rebounds_total: number;
  steals: number;
  blocks: number;
  turnovers: number;
  plus_minus_points: number;
}

// Endpoint-specific Responses
export interface TeamRoster {
  team: Team;
  roster: Player[];
}

export interface PredictionRequest {
  home_team_id: number;
  away_team_id: number;
}

export interface PredictionResponse {
  home_team_id: number;
  away_team_id: number;
  home_win_probability: number;
  predicted_home_score: number;
  predicted_away_score: number;
  confidence_score: number;
}

export interface PlayerProjectionRequest {
  player_id: number;
  opponent_team_id: number;
  home_game: boolean;
}

export interface PlayerProjectionResponse {
  player_id: number;
  projections: Record<string, number>;
  confidence_score: number;
}

export interface PlayerImpactRating {
  person_id: number;
  full_name: string;
  season_year: number;
  primary_position?: string;
  player_impact_rating: number;
  avg_ts_percentage: number;
}

export interface LeagueLeader {
  player: Player;
  team: Team;
  games_played: number;
  avg_points: number;
  avg_assists: number;
  avg_rebounds: number;
}

export interface HeadToHead {
  team_1: Team;
  team_2: Team;
  team_1_wins: number;
  team_2_wins: number;
  total_games: number;
  avg_margin: number;
  last_meeting: string;
}

// Re-export for convenience
export type GamePrediction = PredictionResponse;