import axios, { AxiosInstance } from "axios";

import {
  PaginatedResponse,
  Player,
  Team,
  Game,
  PlayerStats,
  TeamRoster,
  PredictionRequest,
  PredictionResponse,
  PlayerProjectionRequest,
  PlayerProjectionResponse,
  LeagueLeader,
  HeadToHead,
  PlayerImpactRating,
} from "./types";

export default class ApiClient {
  private api: AxiosInstance;

  constructor(baseURL: string) {
    this.api = axios.create({
      baseURL: `${baseURL}/api/v1`,
      headers: {
        "Content-Type": "application/json",
      },
    });
  }

  // Player Endpoints
  searchPlayers = (query: string, limit = 10, offset = 0) =>
    this.api
      .get<PaginatedResponse<Player>>("/players/search", {
        params: { q: query, limit, offset },
      })
      .then((res) => res.data);

  getPlayer = (playerId: number) =>
    this.api.get<Player>(`/players/${playerId}`).then((res) => res.data);

  getPlayerStats = (playerId: number, limit = 10, offset = 0) =>
    this.api
      .get<PaginatedResponse<PlayerStats>>(`/players/${playerId}/stats`, {
        params: { limit, offset },
      })
      .then((res) => res.data);

  // Team Endpoints
  getTeams = (limit = 30, offset = 0) =>
    this.api
      .get<PaginatedResponse<Team>>("/teams", { params: { limit, offset } })
      .then((res) => res.data);

  getTeamRoster = (teamId: number) =>
    this.api.get<TeamRoster>(`/teams/${teamId}/roster`).then((res) => res.data);

  // Game Endpoints
  getTodaysGames = () =>
    this.api.get<Game[]>("/games/today").then((res) => res.data);

  getGame = (gameId: number) =>
    this.api.get<Game>(`/games/${gameId}`).then((res) => res.data);

  // Analytics Endpoints
  getLeagueLeaders = (seasonYear: number, limit = 10) =>
    this.api
      .get<LeagueLeader[]>("/analytics/league-leaders", {
        params: { season_year: seasonYear, limit },
      })
      .then((res) => res.data);

  getHeadToHead = (team1Id: number, team2Id: number) =>
    this.api
      .get<HeadToHead>("/analytics/h2h", {
        params: { team1_id: team1Id, team2_id: team2Id },
      })
      .then((res) => res.data);

  getPlayerImpactRatings = (seasonYear: number, limit = 25) =>
    this.api
      .get<PlayerImpactRating[]>("/analytics/player-impact", {
        params: { season_year: seasonYear, limit },
      })
      .then((res) => res.data);

  // Prediction Endpoints
  predictGame = (homeTeamId: number, awayTeamId: number) =>
    this.api
      .post<PredictionResponse>("/predictions/game", {
        home_team_id: homeTeamId,
        away_team_id: awayTeamId,
      } as PredictionRequest)
      .then((res) => res.data);

  predictPlayerStats = (
    playerId: number,
    opponentTeamId: number,
    homeGame: boolean,
  ) =>
    this.api
      .post<PlayerProjectionResponse>("/predictions/player", {
        player_id: playerId,
        opponent_team_id: opponentTeamId,
        home_game: homeGame,
      } as PlayerProjectionRequest)
      .then((res) => res.data);
}