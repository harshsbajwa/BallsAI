"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@nba/ui/card";
import { useQuery } from "@tanstack/react-query";

import { GamePredictor } from "@/components/game-predictor";
import { PageHeader } from "@/components/page-header";
import { apiClient } from "@/lib/api-client";

interface GamePageProps {
  params: {
    id: string;
  };
}

export default function GamePage({ params }: GamePageProps) {
  const gameId = parseInt(params.id, 10);

  const { data: game, isLoading } = useQuery({
    queryKey: ["game", gameId],
    queryFn: () => apiClient.getGame(gameId),
    enabled: !!gameId,
  });

  if (isLoading) return <p className="container mx-auto p-4">Loading game details...</p>;
  if (!game) return <p className="container mx-auto p-4">Game not found.</p>;

  const isUpcoming = game.home_score === null || game.away_score === null;
  const gameTime = new Date(game.game_date).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <div className="container mx-auto px-4 py-8">
      <PageHeader
        title={`${game.away_team.team_name} at ${game.home_team.team_name}`}
        description={`Game details for ${new Date(game.game_date).toLocaleDateString()}`}
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mt-8">
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle>Game Summary</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex justify-between items-center text-center mb-8">
                <div className="flex-1">
                  <h2 className="text-2xl font-bold">{game.away_team.full_name}</h2>
                  <p className="text-muted-foreground">Away</p>
                </div>
                <div className="px-4">
                  <p className="text-xl text-muted-foreground">VS</p>
                </div>
                <div className="flex-1">
                  <h2 className="text-2xl font-bold">{game.home_team.full_name}</h2>
                  <p className="text-muted-foreground">Home</p>
                </div>
              </div>

              <div className="flex justify-center items-center text-center">
                {isUpcoming ? (
                  <div>
                    <p className="text-4xl font-bold">{gameTime}</p>
                    <p className="text-muted-foreground mt-2">Tip-off Time</p>
                  </div>
                ) : (
                  <div className="flex items-center">
                    <p className="text-6xl font-bold">{game.away_score}</p>
                    <p className="text-4xl font-bold text-muted-foreground mx-8">-</p>
                    <p className="text-6xl font-bold">{game.home_score}</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
        <div className="space-y-8">
            <GamePredictor 
                initialHomeTeamId={String(game.home_team.team_id)}
                initialAwayTeamId={String(game.away_team.team_id)}
            />
        </div>
      </div>
    </div>
  );
}