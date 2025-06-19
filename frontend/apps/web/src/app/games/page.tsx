"use client";

import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
import { GameCard } from "@/components/game-card";
import { PageHeader } from "@/components/page-header";
import { Game } from "@nba/api/types";

export default function GamesPage() {
  const {
    data: todaysGames,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["games", "today"],
    queryFn: () => apiClient.getTodaysGames(),
  });

  return (
    <main className="container mx-auto px-4 py-8">
      <PageHeader
        title="Today's Games"
        description="Scores and schedules for all games happening today."
      />

      {isLoading && <p>Loading games...</p>}
      {error && <p className="text-red-500">Error loading games.</p>}

      {todaysGames?.length === 0 && (
        <div className="text-center py-16">
          <p className="text-lg text-muted-foreground">
            No games scheduled for today.
          </p>
        </div>
      )}

      {todaysGames && todaysGames.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {todaysGames.map((game: Game) => (
            <GameCard key={game.game_id} game={game} />
          ))}
        </div>
      )}
    </main>
  );
}