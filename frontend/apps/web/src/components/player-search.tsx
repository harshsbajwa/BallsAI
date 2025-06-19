"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
import { Input } from "@nba/ui/input";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@nba/ui/card";
import Link from "next/link";
import { Player } from "@nba/api/types";

export function PlayerSearch() {
  const [query, setQuery] = useState("");

  const { data, isLoading } = useQuery({
    queryKey: ["players", "search", query],
    queryFn: () => apiClient.searchPlayers(query, 5),
    enabled: query.length >= 2,
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle>Player Search</CardTitle>
        <CardDescription>Find any player in the league.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <Input
          placeholder="e.g., Stephen Curry"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        {isLoading && <p className="text-sm text-muted-foreground">Searching...</p>}
        {data && query.length >= 2 && (
          <div className="space-y-2">
            {data.items.length > 0 ? (
              data.items.map((player: Player) => (
                <Link
                  key={player.person_id}
                  href={`/players/${player.person_id}`}
                >
                  <div className="p-2 rounded-md hover:bg-accent">
                    <p className="font-semibold">{player.full_name}</p>
                    <p className="text-sm text-muted-foreground">
                      {player.primary_position}
                    </p>
                  </div>
                </Link>
              ))
            ) : (
              <p className="text-sm text-center text-muted-foreground pt-2">
                No results found.
              </p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}