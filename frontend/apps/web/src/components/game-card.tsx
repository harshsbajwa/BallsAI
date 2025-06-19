import { Game } from "@nba/api/types";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@nba/ui/card";
import Link from "next/link";

interface GameCardProps {
  game: Game;
}

export function GameCard({ game }: GameCardProps) {
  const gameTime = new Date(game.game_date).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
  const isUpcoming = game.home_score === null || game.away_score === null;

  return (
    <Link href={`/games/${game.game_id}`} className="block hover:scale-[1.02] transition-transform">
        <Card className="h-full">
        <CardHeader>
            <CardTitle className="flex justify-between items-center">
            <span>{game.away_team.team_abbrev}</span>
            <span className="text-muted-foreground text-sm">@</span>
            <span>{game.home_team.team_abbrev}</span>
            </CardTitle>
            <CardDescription className="flex justify-between items-center">
            <span>{game.away_team.full_name}</span>
            <span>{game.home_team.full_name}</span>
            </CardDescription>
        </CardHeader>
        <CardContent className="flex justify-between items-center">
            <span
            className={`text-4xl font-bold ${
                !isUpcoming && (game.away_score ?? 0) > (game.home_score ?? 0)
                ? "text-foreground"
                : "text-muted-foreground"
            }`}
            >
            {game.away_score ?? "-"}
            </span>
            <span
            className={`text-4xl font-bold ${
                !isUpcoming && (game.home_score ?? 0) > (game.away_score ?? 0)
                ? "text-foreground"
                : "text-muted-foreground"
            }`}
            >
            {game.home_score ?? "-"}
            </span>
        </CardContent>
        <CardFooter>
            <p className="text-sm text-muted-foreground">
            {isUpcoming ? `Today at ${gameTime}` : "Final"}
            </p>
        </CardFooter>
        </Card>
    </Link>
  );
}