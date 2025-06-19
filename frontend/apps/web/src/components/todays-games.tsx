"use client";

import { Game } from "@nba/api/types";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@nba/ui/card";
import { useQuery } from "@tanstack/react-query";

import { apiClient } from "@/lib/api-client";

import { GameCard } from "./game-card";


export function TodaysGames() {
  const { data, isLoading } = useQuery({
    queryKey: ["games", "today"],
    queryFn: () => apiClient.getTodaysGames(),
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle>Today`&apos;`s Slate</CardTitle>
        <CardDescription>
          A look at all the games scheduled for today.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {isLoading && <p>Loading games...</p>}
        {data?.length === 0 && (
          <p className="text-muted-foreground">No games today.</p>
        )}
        {data && data.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {data.map((game: Game) => (
              <GameCard key={game.game_id} game={game} />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}